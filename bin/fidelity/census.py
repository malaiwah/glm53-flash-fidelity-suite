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
                "run the cloud recipe instead: bin/measure-cloud --lane streaming",
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
# Window-major cost model (the REAL engine's schedule)
# --------------------------------------------------------------------------
# stream_score.py has exactly one --stream-mode: window-major.  The layer-outer
# schedule the solver above prices is a HYPOTHESIS no engine implements today,
# so the planner must also price the engine that exists.  Everything here is
# arithmetic over measured constants; anything unmeasured is emitted as null
# with the instruction for measuring it, never as a guess.

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
