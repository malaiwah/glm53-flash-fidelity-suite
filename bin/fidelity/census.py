"""Model census, fit estimation and the memory solver.

This module is the shared, offline-testable heart of BOTH runners.  Everything
here is pure arithmetic over a `Census` plus a `Device`; no network, no torch,
no GPU.  `selftest_fit.py` exercises it against known-answer cases, which is why
it deliberately has no imports beyond the stdlib.

UNITS.  Every byte count is decimal (GB = 1e9), because that is what Hugging
Face reports and what every published figure in this campaign was quoted in.
`gib()` is provided for the places a human expects nvidia-smi's units.

CENSUS PROVENANCE.  The non-routed footprint is derived by SUBTRACTION:

    nonrouted = total_safetensors_bytes - routed_main - routed_mtp

not by summing a hand-written list of tensor shapes.  Subtraction needs only
one cheap HF API call (`?blobs=true`) and it reproduces the independently
measured 19.34 GB figure (which came from range-fetching all 47 non-routed
shard headers).  A shape-summing derivation of the same quantity came out
~1 GB low because GLM5Next's linear-attention layers and the MTP block do not
have the parameter shapes a generic MoE census assumes.  Subtraction cannot
make that class of mistake: anything it fails to classify as routed lands in
non-routed, which is the conservative direction for a fit estimate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

GB = 1_000_000_000.0
GIB = float(1 << 30)


def gb(n_bytes: float) -> float:
    return n_bytes / GB


def gib(n_bytes: float) -> float:
    return n_bytes / GIB


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------


@dataclass
class Census:
    """Decoded-BF16 footprint of a base model, split routed vs non-routed.

    The critical structural fact this type encodes: the decoded VRAM footprint
    is a property of the BASE MODEL, not of the quant's bits-per-weight.  A
    4bpw and an 8bpw artifact of the same base need identical VRAM once
    decoded; bpw only moves download size and disk.  Instance selection is
    therefore driven by `routed_main_bytes`/`nonrouted_bytes` here, and disk by
    the artifact's own on-disk size.
    """

    model_id: str
    revision: Optional[str] = None

    num_layers: int = 0
    first_k_dense: int = 0
    n_routed_experts: int = 0
    experts_per_tok: int = 0
    hidden: int = 0
    moe_inter: int = 0
    dense_inter: int = 0
    vocab: int = 0
    hc_mult: int = 1
    n_mtp: int = 0

    total_bf16_bytes: float = 0.0
    nonrouted_bytes: float = 0.0
    routed_main_bytes: float = 0.0
    routed_mtp_bytes: float = 0.0

    census_source: str = "derived"  # "hf-blobs" | "derived" | "pinned"
    notes: List[str] = field(default_factory=list)

    # ---- geometry helpers -------------------------------------------------

    @property
    def routed_layers(self) -> int:
        return max(0, self.num_layers - self.first_k_dense)

    @property
    def per_expert_params(self) -> int:
        """gate + up + down, each hidden x moe_inter."""
        return 3 * self.hidden * self.moe_inter

    @property
    def per_expert_bf16_bytes(self) -> float:
        return float(self.per_expert_params) * 2.0

    @property
    def per_routed_layer_bytes(self) -> float:
        return float(self.n_routed_experts) * self.per_expert_bf16_bytes

    @property
    def nonrouted_per_layer_bytes(self) -> float:
        n = max(1, self.num_layers + self.n_mtp)
        return self.nonrouted_bytes / n

    @property
    def routed_matrices_per_pass(self) -> int:
        """Number of individual packed matrices decoded in one full forward."""
        return self.routed_layers * self.n_routed_experts * 3

    def logits_bytes(self, ctx: int, dtype_bytes: int = 4) -> float:
        return float(ctx) * float(self.vocab) * float(dtype_bytes)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_id: str,
        config: Dict[str, Any],
        total_safetensors_bytes: Optional[float] = None,
        revision: Optional[str] = None,
    ) -> "Census":
        text = config.get("text_config", config)
        c = cls(
            model_id=model_id,
            revision=revision,
            num_layers=int(text.get("num_hidden_layers", 0)),
            first_k_dense=int(text.get("first_k_dense_replace", 0)),
            n_routed_experts=int(text.get("n_routed_experts", 0)),
            experts_per_tok=int(text.get("num_experts_per_tok", 0)),
            hidden=int(text.get("hidden_size", 0)),
            moe_inter=int(text.get("moe_intermediate_size", 0)),
            dense_inter=int(text.get("intermediate_size", 0)),
            vocab=int(text.get("vocab_size", 0)),
            hc_mult=int(text.get("hc_mult", 1) or 1),
            n_mtp=int(text.get("num_nextn_predict_layers", 0)),
        )
        c.routed_main_bytes = c.routed_layers * c.per_routed_layer_bytes
        c.routed_mtp_bytes = c.n_mtp * c.per_routed_layer_bytes
        if total_safetensors_bytes:
            c.total_bf16_bytes = float(total_safetensors_bytes)
            c.nonrouted_bytes = max(
                0.0,
                c.total_bf16_bytes - c.routed_main_bytes - c.routed_mtp_bytes,
            )
            c.census_source = "hf-blobs"
        else:
            # No blob listing available (offline / --dry-run without network).
            # Fall back to a shape-summed estimate and SAY SO, because it is
            # known to run about a gigabyte light on this architecture.
            c.nonrouted_bytes = c._shape_summed_nonrouted()
            c.total_bf16_bytes = (
                c.nonrouted_bytes + c.routed_main_bytes + c.routed_mtp_bytes
            )
            c.census_source = "derived"
            c.notes.append(
                "non-routed footprint is a shape-summed ESTIMATE (no blob "
                "listing available); it runs ~1 GB light on GLM5Next-family "
                "geometry. Re-run with network access for the exact figure."
            )
        return c

    def _shape_summed_nonrouted(self) -> float:
        """Coarse fallback only.  See the class docstring for why subtraction wins."""
        h, v = self.hidden, self.vocab
        embed = 2.0 * v * h * 2.0                       # embed_tokens + lm_head
        per_layer_attn = 4.0 * h * h                    # q,k,v,o at hidden^2 scale
        shared_expert = 3.0 * h * self.moe_inter
        dense_mlp = 3.0 * h * self.dense_inter
        norms = 8.0 * h
        n_dense = self.first_k_dense
        n_moe = self.routed_layers
        params = (
            (n_dense + n_moe) * (per_layer_attn + norms)
            + n_dense * dense_mlp
            + n_moe * shared_expert
            + self.n_mtp * (per_layer_attn + norms + shared_expert)
        )
        return embed + params * 2.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["routed_layers"] = self.routed_layers
        d["per_expert_params"] = self.per_expert_params
        d["gb"] = {
            "total_bf16": round(gb(self.total_bf16_bytes), 2),
            "nonrouted": round(gb(self.nonrouted_bytes), 2),
            "routed_main": round(gb(self.routed_main_bytes), 2),
            "routed_mtp": round(gb(self.routed_mtp_bytes), 2),
            "per_routed_layer": round(gb(self.per_routed_layer_bytes), 2),
        }
        return d


# GLM-5.3-Flash geometry, pinned so `--dry-run` works with no network at all.
# Values are the authoritative config's, cross-checked against dione_surface.py
# (MAIN_ROUTED_LAYERS = range(3,45), NUM_EXPERTS = 288) and kld_report.py
# (vocab 154880, 25 windows x 2047 positions).
GLM53_FLASH_CONFIG: Dict[str, Any] = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "text_config": {
        "model_type": "glm5_next_text",
        "num_hidden_layers": 45,
        "first_k_dense_replace": 3,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "intermediate_size": 12288,
        "vocab_size": 154880,
        "hc_mult": 4,
        "num_nextn_predict_layers": 1,
    },
}
# zai-org/GLM-5.3-Flash-BF16, published size.  Subtracting the routed census
# from this reproduces the independently range-fetched 19.34 GB non-routed
# figure, which is why this constant is worth pinning.
GLM53_FLASH_BF16_TOTAL_BYTES = 642.7 * GB


def glm53_flash_census(revision: Optional[str] = None) -> Census:
    return Census.from_config(
        "zai-org/GLM-5.3-Flash-BF16",
        GLM53_FLASH_CONFIG,
        total_safetensors_bytes=GLM53_FLASH_BF16_TOTAL_BYTES,
        revision=revision,
    )


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


@dataclass
class Device:
    name: str
    kind: str                    # "cuda" | "mps" | "cpu"
    memory_bytes: float          # per-accelerator budget available to us
    count: int = 1
    unified: bool = False        # Apple Silicon: host RAM and VRAM are one pool
    host_ram_bytes: Optional[float] = None
    note: str = ""

    @property
    def total_bytes(self) -> float:
        return self.memory_bytes * self.count


# Known-answer devices used by the selftest and by --explain.  `memory_bytes`
# is the usable budget, not the marketing number: a 32 GB card cannot hand a
# process 32 GB.
H200 = Device("NVIDIA H200", "cuda", 141 * GB, note="141 GB HBM3e")
H100_80 = Device("NVIDIA H100 80GB", "cuda", 80 * GB)
RTX_PRO6000 = Device("NVIDIA RTX PRO 6000", "cuda", 96 * GB)
RTX_5090 = Device("NVIDIA RTX 5090", "cuda", 32 * GB, note="32 GB GDDR7")
MAC_128 = Device(
    "Apple M-series 128 GB", "mps", 128 * GB, unified=True, host_ram_bytes=128 * GB,
    note="unified memory; the OS will not let one process have all of it",
)
GTX_1650 = Device("NVIDIA GTX 1650", "cuda", 4 * GB, note="deliberately too small")


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------

LANES = ("sealed-ep8", "streaming", "local-mps", "local-cuda-budget")

# Activation headroom for the sealed EP8 lane, in bytes.  Provenance: the
# fp32 logit buffer for one 2048-token window is exactly ctx*vocab*4 =
# 1.269 GB; the rest is attention workspace, the 4-wide hyper-connection
# residual, NCCL buffers and allocator slack.  8 GB total is the figure the
# sealed lane was actually scheduled against on H200.
SEALED_ACTIVATION_BYTES = 8.0 * GB

# Streaming lane peak, OBSERVED not derived: 34-47 GB on one H200 under the
# current schedule.  We size against the top of the observed band times a
# headroom factor, because a peak that was observed once is not a bound.
STREAM_PEAK_OBSERVED_BYTES = 47.0 * GB
STREAM_HEADROOM = 1.35

# Framework floor: CUDA context + torch allocator + cuBLAS/cuDNN workspaces.
# Small, but it is the difference between "fits" and "OOM at layer 41".
FRAMEWORK_OVERHEAD_BYTES = 1.0 * GB
ATTENTION_WORKSPACE_BYTES = 0.5 * GB


@dataclass
class MemoryPlan:
    """A solved local schedule.  See `solve_local` for the invariance property."""

    expert_chunk: int
    window_batch: int
    decode_batch_matrices: int
    buffers: int
    bits: float
    passes: int
    peak_bytes: float
    layer_peak_bytes: float
    head_peak_bytes: float
    breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["peak_gb"] = round(gb(self.peak_bytes), 2)
        d["breakdown_gb"] = {k: round(gb(v), 3) for k, v in self.breakdown.items()}
        return d


def local_peak_bytes(
    census: Census,
    *,
    expert_chunk: int,
    window_batch: int,
    decode_batch_matrices: int,
    buffers: int,
    bits: float,
    ctx: int,
    nonrouted_resident: bool = False,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Peak accelerator bytes for the panel-batched, layer-outer schedule.

    The schedule decodes each expert exactly ONCE for the whole panel instead
    of once per window, at the price of holding the panel's inter-layer state
    resident.  For a 25-window panel that state is ~2.9 GB and the saving is a
    25x cut in decode and weight I/O -- which is the dominant cost, because a
    per-window schedule re-reads the entire checkpoint 25 times.

    Two peaks matter and they are not concurrent, so we take the max:
      * layer_peak -- inside the expert loop, where the decoded chunk lives;
      * head_peak  -- at the lm_head, where one window's fp32 logits live.
    """
    pe_bf16 = census.per_expert_bf16_bytes
    pe_packed = float(census.per_expert_params) * (bits / 8.0)

    chunk_decoded = buffers * expert_chunk * pe_bf16
    chunk_packed = buffers * expert_chunk * pe_packed

    # Decode intermediate: the unpacked bitstream is 4 * P_matrix * bits bytes.
    # Verified against a real decode: bits=6, P=8,388,608 -> 201.3 MB.
    per_matrix_workspace = 4.0 * float(census.hidden * census.moe_inter) * bits
    decode_ws = decode_batch_matrices * per_matrix_workspace

    tok = float(window_batch) * float(ctx)
    residual = tok * census.hidden * census.hc_mult * 2.0   # bf16, hc_mult streams
    collapsed = tok * census.hidden * 2.0                   # bf16 MoE input
    accum = tok * census.hidden * 4.0                       # fp32 MoE accumulator
    panel_state = residual + collapsed + accum

    if nonrouted_resident:
        nonrouted = census.nonrouted_bytes
    else:
        nonrouted = buffers * census.nonrouted_per_layer_bytes

    base = panel_state + nonrouted + ATTENTION_WORKSPACE_BYTES + FRAMEWORK_OVERHEAD_BYTES

    layer_peak = base + chunk_decoded + chunk_packed + decode_ws

    # The head runs one window at a time and the decoded expert chunk is
    # already freed -- but the lm_head WEIGHT must be resident to produce the
    # logits, and on a 154,880-token vocabulary that matrix is as large as the
    # logits themselves (4096 x 154880 x 2 = 1.27 GB each).  Counting only the
    # output buffer understates the floor by a full gigabyte and makes 4 GB
    # cards look viable when they are not.  When the non-routed weights are
    # already resident (unified memory) the lm_head is inside `nonrouted` and
    # must not be counted twice.
    lm_head_weight = 0.0 if nonrouted_resident else float(census.vocab) * census.hidden * 2.0
    head_peak = base + lm_head_weight + census.logits_bytes(ctx, 4)

    breakdown = {
        "panel_state": panel_state,
        "residual_streams": residual,
        "collapsed_moe_input": collapsed,
        "moe_accumulator_fp32": accum,
        "nonrouted": nonrouted,
        "decoded_expert_chunk": chunk_decoded,
        "packed_expert_chunk": chunk_packed,
        "decode_workspace": decode_ws,
        "attention_workspace": ATTENTION_WORKSPACE_BYTES,
        "framework_overhead": FRAMEWORK_OVERHEAD_BYTES,
        "lm_head_weight": lm_head_weight,
        "lm_head_logits_fp32": census.logits_bytes(ctx, 4),
    }
    return max(layer_peak, head_peak), layer_peak, head_peak, breakdown


def solve_local(
    census: Census,
    device: Device,
    *,
    budget_bytes: float,
    bits: float,
    ctx: int = 2048,
    windows: int = 25,
    decode_batch_matrices: int = 4,
    buffers: int = 2,
    nonrouted_resident: Optional[bool] = None,
    fill_fraction: float = 0.85,
) -> Optional[MemoryPlan]:
    """Largest schedule that fits `budget_bytes`, or None if nothing does.

    Search order matters and is deliberate:

      1. Keep the whole panel batched (window_batch = windows) and shrink
         `expert_chunk`.  Shrinking the chunk costs nothing but kernel-launch
         overhead -- decode still happens exactly once per expert per pass.
      2. Only when even expert_chunk=1 will not fit do we shrink
         `window_batch`, because THAT costs a whole extra pass over the
         checkpoint per split: ceil(windows / window_batch) passes.

    INVARIANCE.  Both knobs are numerics-invariant.  Experts are visited in
    strictly ascending order and accumulated sequentially into an fp32
    accumulator, so the result is bit-identical for any (expert_chunk,
    window_batch).  This holds ONLY for a sequential scatter-add; an
    atomicAdd-based scatter would break it, which is why the runner forbids
    one.  `selftest_fit.py` asserts the property holds in the solver, and the
    engine contract requires a bitwise fixture check at the extremes.
    """
    if nonrouted_resident is None:
        nonrouted_resident = bool(device.unified)

    # Target a fraction of the stated budget rather than filling it.  Maxing
    # `expert_chunk` until peak == budget is the wrong trade: decode happens
    # exactly once per expert per pass either way, so a bigger chunk buys
    # almost nothing, while a peak sitting flush against the ceiling turns
    # ordinary allocator fragmentation into an OOM at layer 41.  The budget is
    # still a hard bound; this is how much of it we aim to use.
    target = budget_bytes * max(0.05, min(1.0, fill_fraction))

    def peak(e: int, w: int, b: int) -> Tuple[float, float, float, Dict[str, float]]:
        return local_peak_bytes(
            census,
            expert_chunk=e,
            window_batch=w,
            decode_batch_matrices=decode_batch_matrices,
            buffers=b,
            bits=bits,
            ctx=ctx,
            nonrouted_resident=nonrouted_resident,
        )

    for w in _window_ladder(windows):
        for b in (buffers, 1) if buffers > 1 else (1,):
            lo, hi = 1, census.n_routed_experts
            best: Optional[int] = None
            while lo <= hi:
                mid = (lo + hi) // 2
                p, _, _, _ = peak(mid, w, b)
                if p <= target:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best is None:
                # Nothing fits the soft target.  Before giving up on this
                # (w, b), see whether the smallest chunk fits the HARD budget:
                # a run that is tight is better than no run at all, and we say
                # so in the plan rather than silently pretending it is roomy.
                p1, _, _, _ = peak(1, w, b)
                if p1 <= budget_bytes:
                    best = 1
            if best is not None:
                total, layer_p, head_p, br = peak(best, w, b)
                return MemoryPlan(
                    expert_chunk=best,
                    window_batch=w,
                    decode_batch_matrices=decode_batch_matrices,
                    buffers=b,
                    bits=bits,
                    passes=math.ceil(windows / w),
                    peak_bytes=total,
                    layer_peak_bytes=layer_p,
                    head_peak_bytes=head_p,
                    breakdown=br,
                )
            # expert_chunk=1 did not fit at this (w, b); try fewer buffers,
            # then fewer windows.
    return None


def _window_ladder(windows: int) -> List[int]:
    ladder, w = [], windows
    while w >= 1:
        ladder.append(w)
        if w == 1:
            break
        w = max(1, w // 2)
    return ladder


def minimum_viable_budget(census: Census, *, bits: float, ctx: int = 2048) -> float:
    """The smallest budget under which ANY local schedule runs.

    This is what a refusal message must quote.  Telling someone "it does not
    fit" without telling them what would fit is not advice.
    """
    total, _, _, _ = local_peak_bytes(
        census,
        expert_chunk=1,
        window_batch=1,
        decode_batch_matrices=1,
        buffers=1,
        bits=bits,
        ctx=ctx,
        nonrouted_resident=False,
    )
    return total


# --------------------------------------------------------------------------
# Cloud / multi-GPU lane requirements
# --------------------------------------------------------------------------


@dataclass
class LaneRequirement:
    lane: str
    gpus: int
    ep_size: int
    per_gpu_bytes: float
    components: Dict[str, float]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_gpu_gb"] = round(gb(self.per_gpu_bytes), 1)
        d["components_gb"] = {k: round(gb(v), 2) for k, v in self.components.items()}
        return d


def lane_requirement(census: Census, lane: str, *, ctx: int = 2048) -> LaneRequirement:
    if lane == "sealed-ep8":
        ep = 8
        resident = census.nonrouted_bytes + census.routed_main_bytes / ep
        per_gpu = resident + SEALED_ACTIVATION_BYTES
        return LaneRequirement(
            lane=lane,
            gpus=ep,
            ep_size=ep,
            per_gpu_bytes=per_gpu,
            components={
                "nonrouted_full_replica": census.nonrouted_bytes,
                "routed_main_shard": census.routed_main_bytes / ep,
                "activations": SEALED_ACTIVATION_BYTES,
            },
            rationale=(
                "EP8 replicates every non-routed tensor on all 8 ranks and shards "
                "the routed experts 8 ways. The MTP block's routed experts are not "
                "executed during capture (mtp_standard_logits_executed=false) and "
                "are excluded."
            ),
        )
    if lane == "streaming":
        per_gpu = STREAM_PEAK_OBSERVED_BYTES * STREAM_HEADROOM
        return LaneRequirement(
            lane=lane,
            gpus=1,
            ep_size=1,
            per_gpu_bytes=per_gpu,
            components={
                "observed_peak": STREAM_PEAK_OBSERVED_BYTES,
                "headroom": per_gpu - STREAM_PEAK_OBSERVED_BYTES,
            },
            rationale=(
                "OBSERVED 34-47 GB on one H200 under the current schedule, not "
                "derived. Sized against the top of the band x%.2f, because a peak "
                "seen once is not a bound." % STREAM_HEADROOM
            ),
        )
    raise ValueError(
        "lane %r has no cloud requirement; local lanes are solved by solve_local()"
        % (lane,)
    )


@dataclass
class Verdict:
    ok: bool
    reason: str
    advice: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


def check_device(
    census: Census,
    device: Device,
    lane: str,
    *,
    bits: float = 4.0,
    budget_bytes: Optional[float] = None,
    ctx: int = 2048,
    windows: int = 25,
) -> Verdict:
    """Fit check with refuse-WITH-ADVICE semantics.

    A refusal that does not name the next thing to try is a dead end, so every
    `ok=False` path here populates `advice`.
    """
    if lane in ("sealed-ep8", "streaming"):
        req = lane_requirement(census, lane, ctx=ctx)
        if device.count < req.gpus:
            return Verdict(
                False,
                "lane %s needs %d GPUs; device declares %d"
                % (lane, req.gpus, device.count),
                advice=[
                    "--lane streaming runs on a single GPU"
                    if lane == "sealed-ep8"
                    else "request more GPUs",
                ],
                detail={"requirement": req.to_dict()},
            )
        if device.memory_bytes < req.per_gpu_bytes:
            short = req.per_gpu_bytes - device.memory_bytes
            advice = [
                "%s has %.0f GB/GPU; lane %s needs >=%.0f GB/GPU (short %.0f GB)"
                % (device.name, gb(device.memory_bytes), lane,
                   gb(req.per_gpu_bytes), gb(short)),
            ]
            if lane == "sealed-ep8":
                stream = lane_requirement(census, "streaming", ctx=ctx)
                advice.append(
                    "--lane streaming needs only >=%.0f GB on ONE GPU"
                    % gb(stream.per_gpu_bytes)
                )
                advice.append(
                    "or pick a larger GPU: this census needs %.0f GB/GPU at EP8"
                    % gb(req.per_gpu_bytes)
                )
            else:
                mv = minimum_viable_budget(census, bits=bits, ctx=ctx)
                advice.append(
                    "the local panel-batched lanes run from %.1f GB upward "
                    "(bin/measure-local --vram-budget)" % gb(mv)
                )
            return Verdict(False, "insufficient VRAM per GPU", advice=advice,
                           detail={"requirement": req.to_dict()})
        return Verdict(
            True,
            "fits: %.0f GB/GPU available, %.0f GB/GPU required"
            % (gb(device.memory_bytes), gb(req.per_gpu_bytes)),
            detail={"requirement": req.to_dict()},
        )

    # Local lanes.
    if budget_bytes is None:
        budget_bytes = default_budget(device)
    plan = solve_local(
        census, device, budget_bytes=budget_bytes, bits=bits, ctx=ctx, windows=windows
    )
    if plan is None:
        mv = minimum_viable_budget(census, bits=bits, ctx=ctx)
        return Verdict(
            False,
            "no schedule fits a %.1f GB budget" % gb(budget_bytes),
            advice=[
                "minimum viable budget for this model at %g bpw is %.1f GB "
                "(expert_chunk=1, window_batch=1, single buffer)" % (bits, gb(mv)),
                "that floor is set by the lm_head step, not by the experts: the "
                "lm_head weight (%.2f GB) and one window of fp32 logits "
                "(%.2f GB) must be resident together. No memory knob goes "
                "below it, because neither term depends on expert_chunk or "
                "window_batch."
                % (gb(float(census.vocab) * census.hidden * 2.0),
                   gb(census.logits_bytes(ctx, 4))),
                "run the cloud recipe instead: docs/THIRD-PARTY-QUICKSTART.md section 3b",
            ],
            detail={"minimum_viable_budget_bytes": mv},
        )
    return Verdict(
        True,
        "fits: peak %.1f GB of a %.1f GB budget"
        % (gb(plan.peak_bytes), gb(budget_bytes)),
        detail={"plan": plan.to_dict()},
    )


def default_budget(device: Device) -> float:
    """What we will actually ask a device for, absent an explicit --vram-budget.

    Discrete cards get 90% of the card; unified-memory Macs get 70% of system
    RAM, because on Apple Silicon the same pool is holding the OS, the page
    cache for a 200 GB mmap'd checkpoint, and the compositor.
    """
    if device.unified:
        return device.memory_bytes * 0.70
    return device.memory_bytes * 0.90


# --------------------------------------------------------------------------
# Disk / RAM requirements
# --------------------------------------------------------------------------


@dataclass
class StorageNeed:
    artifact_bytes: float
    panel_bytes: float
    student_logits_bytes: float
    toolchain_bytes: float
    slack_fraction: float
    # Two cold runs hold BOTH their fp32 student logit trees on disk before
    # the report is sealed (~2x the panel bytes), whether or not the caller
    # keeps them afterwards.  Sizing the filesystem without this transient is
    # lesson 31 (disk-full at window 19 of run 2).
    transient_student_logits_bytes: float = 0.0

    @property
    def total_bytes(self) -> float:
        raw = (
            self.artifact_bytes
            + self.panel_bytes
            + max(self.student_logits_bytes, self.transient_student_logits_bytes)
            + self.toolchain_bytes
        )
        return raw * (1.0 + self.slack_fraction)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_gb"] = round(gb(self.total_bytes), 1)
        return d


def storage_need(
    *,
    artifact_bytes: float,
    panel_bytes: float,
    keep_student_logits: bool,
    toolchain_bytes: float = 40 * GB,
    slack_fraction: float = 0.15,
    cold_runs: int = 2,
    extra_bytes: float = 0.0,
) -> StorageNeed:
    return StorageNeed(
        artifact_bytes=artifact_bytes + extra_bytes,
        panel_bytes=panel_bytes,
        student_logits_bytes=panel_bytes * cold_runs if keep_student_logits else 0.0,
        transient_student_logits_bytes=panel_bytes * cold_runs,
        toolchain_bytes=toolchain_bytes,
        slack_fraction=slack_fraction,
    )


def round_up_storage_gb(n_bytes: float, granularity_gb: int = 100) -> int:
    need = gb(n_bytes)
    return int(math.ceil(need / granularity_gb) * granularity_gb)


# --------------------------------------------------------------------------
# Window-major cost model (the streaming lane's schedule)
# --------------------------------------------------------------------------
# stream_score.py has exactly one --stream-mode: window-major, so the local
# streaming lanes must be priced by it.  The panel-batched layer-outer solver
# above prices a schedule stream_score does NOT run; the layer-outer engine
# that does exist (engines/tools/layer_outer.py under hf_capture.py) never
# batches windows and is priced by `layer_outer_plan` below.  Everything here
# is arithmetic over measured constants; anything unmeasured is emitted as
# null with the instruction for measuring it, never as a guess.

SCORING_MS_PER_POSITION_CPU = 0.15   # MEASURED on the M4 Max (0.144-0.164 ms)
LM_HEAD_TFLOP_PER_WINDOW = 2.60      # 2 * 2047 * hidden(4096) * vocab(154880) / 1e12


def window_major_cost(
    census: Census,
    *,
    windows: int = 25,
    positions_per_window: int = 2047,
    ms_per_matrix: float,
    decode_cache: str = "none",
    budget_bytes: Optional[float] = None,
    disk_gb_per_s: float = 5.5,
    trunk_seconds_per_window: Optional[float] = None,
) -> Dict[str, Any]:
    """Price a full panel pass of the REAL window-major engine.

    ms_per_matrix comes from the caller's micro-benchmark (16-20 MPS / 53-57
    CPU on the M4 Max); trunk_seconds_per_window is None unless someone has
    MEASURED it on this device class -- the KDA/MPS forward speed is the open
    question, and this function refuses to invent it.
    disk_gb_per_s defaults to 5.5 (Apple internal NVMe class) and is a
    parameter precisely because 'measure before assuming' applies to disks
    too (the floor box's CephFS did 0.9-1.05).
    """
    if decode_cache not in ("none", "ram", "disk"):
        raise ValueError("decode_cache must be none|ram|disk")
    matrices = census.routed_matrices_per_pass          # 42*288*3 = 36,288
    decode_pass_s = matrices * ms_per_matrix / 1000.0
    layer_slab = census.per_routed_layer_bytes          # ~14.50 GB
    cached_layers = 0
    if decode_cache == "ram":
        if budget_bytes is None:
            raise ValueError("decode_cache=ram needs budget_bytes")
        cached_layers = min(census.routed_layers,
                            int((0.8 * budget_bytes) // layer_slab))
    if decode_cache == "none":
        decode_pass_equivalents = float(windows)
        disk_reread_s = 0.0
    elif decode_cache == "ram":
        fraction_uncached = 1.0 - cached_layers / float(census.routed_layers)
        decode_pass_equivalents = 1.0 + (windows - 1) * fraction_uncached
        disk_reread_s = 0.0
    else:  # disk: decode once, re-read the decoded bf16 surface per window
        decode_pass_equivalents = 1.0
        disk_reread_s = windows * census.routed_main_bytes / (disk_gb_per_s * GB)
    decode_total_s = decode_pass_equivalents * decode_pass_s
    scoring_s = windows * positions_per_window * SCORING_MS_PER_POSITION_CPU / 1000.0
    trunk_total_s = (None if trunk_seconds_per_window is None
                     else trunk_seconds_per_window * windows)
    total_known_s = decode_total_s + disk_reread_s + scoring_s + (trunk_total_s or 0.0)
    return {
        "stream_mode": "window-major (the engine's only mode)",
        "matrices_per_pass": matrices,
        "ms_per_matrix": ms_per_matrix,
        "decode_seconds_per_pass": decode_pass_s,
        "decode_cache": decode_cache,
        "cached_layers": cached_layers,
        "decode_pass_equivalents": decode_pass_equivalents,
        "decode_seconds_total": decode_total_s,
        "disk_reread_seconds_total": disk_reread_s,
        "disk_gb_per_s_assumed": (disk_gb_per_s if decode_cache == "disk" else None),
        "trunk_seconds_per_window": trunk_seconds_per_window,
        "trunk_seconds_total": trunk_total_s,
        "trunk_note": (None if trunk_seconds_per_window is not None else
                       "UNMEASURED on this device class: 34 of 45 layers are "
                       "Kimi-Delta linear attention with Triton/CUDA-only fast "
                       "paths. Measure via `bin/measure-local --fixture fetch` "
                       "(fixture-scale L1.c timing), then one real window; "
                       "never assume."),
        "lm_head_tflop_per_window": LM_HEAD_TFLOP_PER_WINDOW,
        "scoring_seconds_total": scoring_s,
        "scoring_note": "fp64 KLD scoring is 0.15 ms/position on CPU (measured) "
                        "-- ~8 s/panel: scoring never motivates sampling; "
                        "position sampling is a storage/teacher-bandwidth knob",
        "total_known_seconds": total_known_s,
        "total_is_lower_bound": trunk_seconds_per_window is None,
    }


# --------------------------------------------------------------------------
# Layer-outer capture plan (hf_capture.py --schedule layer-outer, the engine
# that exists for native-bf16 / fp8-block / exl3hf surfaces)
# --------------------------------------------------------------------------
# docs/LAYER-OUTER.md section 8.1 arithmetic:
#
#     peak ~= resident(embed + lm_head + final norm) + largest layer
#             + carried state + epilogue logits + workspace + load transient
#
# The per-layer terms are derived from config.json for the model_types whose
# arithmetic has been checked against a real checkpoint to the byte
# (`LayerGeometry.provenance` says which check).  Anything else refuses: a
# geometry that is guessed is a plan that says "fits" for a card that OOMs.

LAYER_OUTER_SURFACES = ("native-bf16", "fp8-block", "exl3hf")

# Within-layer activations/workspace at hidden 6144, 64 heads, ctx 2048:
# LAYER-OUTER.md 8.1 budgets 2.0-3.0 GB; the top of the band is used.
LAYER_OUTER_WORKSPACE_BYTES = 3.0 * GB

# Measured on one H200 SXM (RunPod US-NC-1) by hf_capture.py --schedule
# layer-outer over the 25 x 2048 corpus5x5 panel, `{"stage": "peak_memory"}`
# lines of the sealed pod logs (quoted in review-efficiency.md section 2 and
# review-local-usability.md, 2026-09-05).  Decimal GB.  The reserved figure
# is what the card must actually have: torch's allocator held it.
GLM53_LAYER_OUTER_MEASURED = {
    "geometry": "glm_moe_dsa 78L / hidden 6144 / 256 experts (GLM-5.3)",
    "device": "NVIDIA H200 SXM 141 GB",
    # The loader these peaks belong to.  A changed loader (direct expert fill,
    # chunked materialisation) needs a NEW measured peak here; the floor below
    # moves on evidence, never on a code change.
    "measured_on": "2026-09-04/05",
    "loader": "transformers converter stack/concatenate materialisation + "
              "trellis decoded dict held on device (pre direct-fill)",
    "peak_resident_weight_bytes": 23_561_229_056,     # 3.81 GB + one 19.76 GB layer
    "native-bf16": {"allocated_bytes": 37.530 * GB, "reserved_bytes": 57.078 * GB,
                    "run": "glm53-resume4 cold run 2, zai-org/GLM-5.3-BF16"},
    "fp8-block": {"allocated_bytes": 37.530 * GB, "reserved_bytes": 57.087 * GB,
                  "run": "glm53-fp8 cold run 1, zai-org/GLM-5.3"},
    "exl3hf": {"allocated_bytes": 56.859 * GB, "reserved_bytes": 58.139 * GB,
               "run": "exl3-wrld11 cold run 1, wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"},
}

# The smallest card the GLM-5.3 geometry runs on TODAY: the transformers
# converter materialises one whole layer's routed experts (19.33 GB) before
# fusing them, and the trellis path additionally holds the decoded per-expert
# dict on device until the layer is fused -- so the measured peaks above are
# 57-58 GB reserved whatever the surface.  The chunked loader that would take
# the peak to ~28 GB (LAYER-OUTER.md 8.1) is not built.  64 GB is the measured
# reserved peak plus the headroom a non-H200 allocator needs; below it the
# planner refuses rather than promising a run that dies at layer 3.
GLM53_CLASS_MIN_DEVICE_BYTES = 64.0 * GB


class GeometryUnknown(ValueError):
    """config.json names an architecture whose per-layer arithmetic is unverified."""


@dataclass
class LayerGeometry:
    """Decoded-bf16 byte census of a model split the way layer-outer streams it.

    `largest_layer_bytes` is the whole decoder layer -- attention, norms,
    indexer, shared expert, router AND routed experts -- because that is the
    unit the engine holds resident (LAYER-OUTER.md section 1).  Dense models
    have `routed_layer_bytes` 0.
    """

    model_type: str
    num_layers: int
    hidden: int
    vocab: int
    n_routed_experts: int
    resident_bytes: float          # embed_tokens + lm_head + final norm
    largest_layer_bytes: float
    routed_layer_bytes: float      # one sparse layer's routed experts, bf16
    total_bf16_bytes: float        # whole checkpoint incl. MTP, for the KAT
    carries_topk_indices: bool     # DSA indexer carries int64 top-k between layers
    index_topk: int
    provenance: str                # "exact: ..." | "averaged: ..."
    notes: List[str] = field(default_factory=list)

    @property
    def geometry_label(self) -> str:
        return "%s %dL / hidden %d / %d experts" % (
            self.model_type, self.num_layers, self.hidden, self.n_routed_experts)

    @property
    def is_glm53_class(self) -> bool:
        return (self.model_type == "glm_moe_dsa" and self.num_layers == 78
                and self.hidden == 6144 and self.n_routed_experts == 256)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["geometry_label"] = self.geometry_label
        d["gb"] = {
            "resident": round(gb(self.resident_bytes), 3),
            "largest_layer": round(gb(self.largest_layer_bytes), 3),
            "routed_layer": round(gb(self.routed_layer_bytes), 3),
            "total_bf16": round(gb(self.total_bf16_bytes), 3),
        }
        return d


def _glm_moe_dsa_geometry(text: Dict[str, Any]) -> LayerGeometry:
    """GLM-5.3's architecture.  Every term below reconciles to the published
    safetensors index: total 1,506,659,919,872 B (delta 0), per-layer 0.747 /
    18.381 / 18.398 / 18.539 GiB (docs/GLM53-ROOT-FEASIBILITY.md section 2)."""
    h = int(text["hidden_size"])
    v = int(text["vocab_size"])
    layers = int(text["num_hidden_layers"])
    heads = int(text["num_attention_heads"])
    q_lora = int(text["q_lora_rank"])
    kv_lora = int(text["kv_lora_rank"])
    qk_nope = int(text["qk_nope_head_dim"])
    qk_rope = int(text["qk_rope_head_dim"])
    qk_head = int(text.get("qk_head_dim", qk_nope + qk_rope))
    v_head = int(text["v_head_dim"])
    inter = int(text["intermediate_size"])
    moe_inter = int(text["moe_intermediate_size"])
    experts = int(text["n_routed_experts"])
    shared = int(text.get("n_shared_experts", 0))
    idx_heads = int(text["index_n_heads"])
    idx_dim = int(text["index_head_dim"])
    mlp_types = list(text.get("mlp_layer_types") or [])
    indexer_types = list(text.get("indexer_types") or [])
    n_mtp = int(text.get("num_nextn_predict_layers", 0))
    if len(mlp_types) != layers or len(indexer_types) != layers:
        raise GeometryUnknown(
            "glm_moe_dsa config lists %d mlp_layer_types and %d indexer_types for "
            "%d layers; the census needs one of each per layer"
            % (len(mlp_types), len(indexer_types), layers))

    attn = (h * q_lora + q_lora                       # q_a_proj + q_a_layernorm
            + q_lora * heads * qk_head                # q_b_proj
            + h * (kv_lora + qk_rope) + kv_lora       # kv_a_proj_with_mqa + kv_a_layernorm
            + kv_lora * heads * (qk_nope + v_head)    # kv_b_proj
            + heads * v_head * h)                     # o_proj
    indexer_full = (q_lora * idx_heads * idx_dim      # wq_b
                    + h * idx_dim + 2 * idx_dim       # wk + k_norm (weight, bias)
                    + h * idx_heads)                  # weights_proj
    norms = 2 * h
    dense_mlp = 3 * h * inter
    routed = experts * 3 * h * moe_inter
    # e_score_correction_bias (one per expert) is stored fp32 in the release:
    # 2 extra bytes per expert per sparse layer over the bf16 count.
    shared_gate = shared * 3 * h * moe_inter + experts * h + experts
    bias_fp32_extra = experts * 2.0

    def layer_params(kind: str, indexer: str) -> int:
        p = attn + norms + (indexer_full if indexer == "full" else 0)
        return p + (dense_mlp if kind == "dense" else routed + shared_gate)

    per_layer = [layer_params(k, i) for k, i in zip(mlp_types, indexer_types)]
    resident = 2 * v * h + h
    # MTP layer: a sparse layer with a full indexer plus eh_proj and 3 norms.
    mtp = n_mtp * (layer_params("sparse", "full") + 2 * h * h + 3 * h)
    n_sparse = sum(1 for kind in mlp_types if kind == "sparse") + n_mtp
    total = (sum(per_layer) + resident + mtp) * 2.0 + n_sparse * bias_fp32_extra
    return LayerGeometry(
        model_type="glm_moe_dsa", num_layers=layers, hidden=h, vocab=v,
        n_routed_experts=experts,
        resident_bytes=resident * 2.0,
        largest_layer_bytes=max(per_layer) * 2.0 + bias_fp32_extra,
        routed_layer_bytes=routed * 2.0,
        total_bf16_bytes=total,
        carries_topk_indices=True,
        index_topk=int(text.get("index_topk", 0)),
        provenance="exact: per-layer shapes from config.json; reconcile to the "
                   "published safetensors index with delta 0 "
                   "(docs/GLM53-ROOT-FEASIBILITY.md section 2)",
    )


def _dense_qwen3_geometry(text: Dict[str, Any]) -> LayerGeometry:
    """Qwen3 dense: GQA attention with q_norm/k_norm, no biases, SwiGLU MLP.
    Reconciles to Qwen/Qwen3-8B's index total 16,381,470,720 B (delta 0)."""
    h = int(text["hidden_size"])
    v = int(text["vocab_size"])
    layers = int(text["num_hidden_layers"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text.get("head_dim") or h // heads)
    inter = int(text["intermediate_size"])
    tied = bool(text.get("tie_word_embeddings", False))
    attn = h * heads * head_dim + 2 * h * kv_heads * head_dim + heads * head_dim * h + 2 * head_dim
    layer = attn + 3 * h * inter + 2 * h
    resident = (1 if tied else 2) * v * h + h
    return LayerGeometry(
        model_type="qwen3", num_layers=layers, hidden=h, vocab=v,
        n_routed_experts=0,
        resident_bytes=resident * 2.0,
        largest_layer_bytes=layer * 2.0,
        routed_layer_bytes=0.0,
        total_bf16_bytes=(layers * layer + resident) * 2.0,
        carries_topk_indices=False, index_topk=0,
        provenance="exact: per-layer shapes from config.json; reconcile to "
                   "Qwen/Qwen3-8B's safetensors index total with delta 0",
    )


def _glm5_next_geometry(text: Dict[str, Any]) -> LayerGeometry:
    """GLM-5.3-Flash.  The routed set is exact; the non-routed set mixes
    Kimi-Delta linear attention, MLA, hyper-connections and an MTP block whose
    shapes a generic census gets ~1 GB wrong (module docstring), so it is taken
    from the pinned blob-subtraction census and AVERAGED over the layers.  The
    plan says so; it is only ever a few hundred MB per layer."""
    census = glm53_flash_census()
    if (int(text.get("num_hidden_layers", 0)) != census.num_layers
            or int(text.get("hidden_size", 0)) != census.hidden
            or int(text.get("n_routed_experts", 0)) != census.n_routed_experts
            or int(text.get("moe_intermediate_size", 0)) != census.moe_inter
            or int(text.get("vocab_size", 0)) != census.vocab):
        raise GeometryUnknown(
            "glm5_next geometry differs from the pinned GLM-5.3-Flash census "
            "(%dL / hidden %d / %d experts); no verified per-layer arithmetic "
            "exists for it" % (int(text.get("num_hidden_layers", 0)),
                               int(text.get("hidden_size", 0)),
                               int(text.get("n_routed_experts", 0))))
    resident = (2.0 * census.vocab * census.hidden + census.hidden) * 2.0
    nonrouted_layers = census.num_layers + census.n_mtp
    per_layer_nonrouted = (census.nonrouted_bytes - resident) / nonrouted_layers
    return LayerGeometry(
        model_type="glm5_next", num_layers=census.num_layers, hidden=census.hidden,
        vocab=census.vocab, n_routed_experts=census.n_routed_experts,
        resident_bytes=resident,
        largest_layer_bytes=census.per_routed_layer_bytes + per_layer_nonrouted,
        routed_layer_bytes=census.per_routed_layer_bytes,
        total_bf16_bytes=census.total_bf16_bytes,
        carries_topk_indices=True,
        index_topk=int(text.get("index_topk", 0)),
        provenance="averaged: routed experts exact from config.json; non-routed "
                   "per-layer share is the pinned blob-subtraction census "
                   "(19.34 GB) spread evenly over %d layers" % nonrouted_layers,
        notes=["the non-routed per-layer term is an average, not this layer's "
               "shapes; it is under 0.4 GB against a 14.5 GB routed set"],
    )


_GEOMETRIES = {
    "glm_moe_dsa": _glm_moe_dsa_geometry,
    "qwen3": _dense_qwen3_geometry,
    "glm5_next": _glm5_next_geometry,
    "glm5_next_text": _glm5_next_geometry,
}


def layer_geometry(config: Dict[str, Any]) -> LayerGeometry:
    """Per-layer bf16 census from a config.json, or GeometryUnknown.

    Only model_types whose arithmetic reconciled to a real checkpoint to the
    byte are accepted; the refusal names the type so the next one can be
    added WITH its check, never guessed.
    """
    text = config.get("text_config", config)
    model_type = str(text.get("model_type") or config.get("model_type") or "")
    builder = _GEOMETRIES.get(model_type)
    if builder is None:
        raise GeometryUnknown(
            "no verified per-layer census for model_type %r (verified: %s); add "
            "its arithmetic to fidelity/census.py and reconcile it to the "
            "checkpoint's safetensors index before planning with it"
            % (model_type, ", ".join(sorted(_GEOMETRIES))))
    try:
        return builder(text)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, GeometryUnknown):
            raise
        raise GeometryUnknown(
            "config.json for model_type %r lacks a field the census needs: %s"
            % (model_type, exc))


@dataclass
class LayerOuterPlan:
    """The layer-outer capture's device footprint, with its refusal decision."""

    surface: str
    geometry: LayerGeometry
    ctx: int
    windows: int
    breakdown: Dict[str, float]
    modelled_peak_bytes: float
    measured: Optional[Dict[str, Any]]     # H200 anchor when the geometry has one
    required_device_bytes: float
    device_bytes: float
    fits: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["geometry"] = self.geometry.to_dict()
        d["breakdown_gb"] = {k: round(gb(v), 3) for k, v in self.breakdown.items()}
        d["modelled_peak_gb"] = round(gb(self.modelled_peak_bytes), 2)
        d["required_device_gb"] = round(gb(self.required_device_bytes), 2)
        d["engine"] = "engines/tools/hf_capture.py --schedule layer-outer"
        return d


def layer_outer_plan(
    geometry: LayerGeometry,
    *,
    surface: str,
    device: Device,
    ctx: int = 2048,
    windows: int = 25,
) -> LayerOuterPlan:
    """Price hf_capture --schedule layer-outer for `surface` on `device`.

    Windows are never batched (LAYER-OUTER.md section 1), so the carried
    state is one hidden row block per window plus the DSA top-k indices, and
    the epilogue logits are ONE window's.  The load transient is the term the
    H200 runs measured: the transformers converter stacks a layer's gate and
    up experts (2/3 of the routed set) before fusing them, and the trellis
    path keeps the whole decoded routed set on device until the fuse.
    """
    if surface not in LAYER_OUTER_SURFACES:
        raise ValueError("layer-outer reads %s, not %r"
                         % ("/".join(LAYER_OUTER_SURFACES), surface))
    g = geometry
    carried = (windows + 1) * ctx * g.hidden * 2.0
    if g.carries_topk_indices and g.index_topk:
        carried += (windows + 1) * ctx * min(g.index_topk, ctx) * 8.0   # int64
    logits = float(ctx) * g.vocab * 2.0                                # bf16 epilogue
    converter = (2.0 / 3.0) * g.routed_layer_bytes
    decoded = 0.0
    if surface == "exl3hf":
        decoded = g.routed_layer_bytes if g.routed_layer_bytes else g.largest_layer_bytes
    breakdown = {
        "resident_weights": g.resident_bytes,
        "largest_layer": g.largest_layer_bytes,
        "carried_state": carried,
        "epilogue_logits_bf16": logits,
        "workspace": LAYER_OUTER_WORKSPACE_BYTES,
        "framework_overhead": FRAMEWORK_OVERHEAD_BYTES,
        "converter_transient": converter,
        "trellis_decoded_layer": decoded,
    }
    modelled = sum(breakdown.values())
    measured = None
    if g.is_glm53_class:
        measured = dict(GLM53_LAYER_OUTER_MEASURED[surface],
                        device=GLM53_LAYER_OUTER_MEASURED["device"],
                        measured_on=GLM53_LAYER_OUTER_MEASURED["measured_on"],
                        loader=GLM53_LAYER_OUTER_MEASURED["loader"],
                        peak_resident_weight_bytes=(
                            GLM53_LAYER_OUTER_MEASURED["peak_resident_weight_bytes"]))
        required = GLM53_CLASS_MIN_DEVICE_BYTES
        fits = device.memory_bytes >= required
        reason = (
            "GLM-5.3-class layer-outer capture needs a >= %d GB device as of the "
            "%s pod runs: measured on %s, %s allocated / %s reserved (%s); the transformers "
            "converter materialises one layer's %s of routed experts before "
            "fusing them%s, and the chunked loader that would take the peak to "
            "~28 GB (docs/LAYER-OUTER.md 8.1) is not built"
            % (int(gb(required)), GLM53_LAYER_OUTER_MEASURED["measured_on"],
               measured["device"],
               "%.2f GB" % gb(measured["allocated_bytes"]),
               "%.2f GB" % gb(measured["reserved_bytes"]), measured["run"],
               "%.2f GB" % gb(g.routed_layer_bytes),
               (" and the trellis path holds the decoded routed set on device "
                "until the fuse" if surface == "exl3hf" else "")))
    else:
        budget = default_budget(device)
        required = modelled / (budget / device.memory_bytes)
        fits = modelled <= budget
        reason = ("modelled peak %.2f GB against a %.2f GB budget (%s of %s); "
                  "geometry %s -- %s"
                  % (gb(modelled), gb(budget),
                     "70%" if device.unified else "90%",
                     "%.0f GB" % gb(device.memory_bytes), g.geometry_label,
                     g.provenance))
    return LayerOuterPlan(
        surface=surface, geometry=g, ctx=ctx, windows=windows,
        breakdown=breakdown, modelled_peak_bytes=modelled, measured=measured,
        required_device_bytes=required, device_bytes=device.memory_bytes,
        fits=fits, reason=reason)
