"""sitecustomize: pin every Triton @autotune to a single config for
launch-deterministic inference.

Bind-mount into the container's site-packages, e.g.:
    -v pin_autotune_sitecustomize.py:/usr/local/lib/python3.12/dist-packages/sitecustomize.py:ro

Rationale: Triton's autotuner selects kernel configs by per-process timing
benchmarks, so separate engine launches can pick different winners; for
reduction-splitting configs (e.g. the FLA/KDA chunk kernels' BK loops) that
changes float accumulation order and yields bitwise-different outputs per
launch (fla-org/flash-linear-attention#945, triton-lang/triton#9368).
Pinning to one config removes the per-launch selection entirely. We prefer a
num_warps=2 config when present (the variants fla#945 measured as bitwise
deterministic), else the first config.
"""


def _pin_triton_autotune():
    import triton.runtime.autotuner as _at

    _orig_init = _at.Autotuner.__init__

    def _pick_one(configs):
        try:
            cfgs = list(configs)
        except TypeError:
            return configs
        if len(cfgs) <= 1:
            return cfgs
        preferred = [c for c in cfgs if getattr(c, "num_warps", None) == 2]
        return [preferred[0] if preferred else cfgs[0]]

    def _patched_init(self, *args, **kwargs):
        args = list(args)
        if "configs" in kwargs:
            kwargs["configs"] = _pick_one(kwargs["configs"])
        elif len(args) >= 3:
            args[2] = _pick_one(args[2])
        return _orig_init(self, *args, **kwargs)

    _at.Autotuner.__init__ = _patched_init
    print("[pin_autotune] Triton autotune pinned to a single config "
          "(prefer num_warps=2) for launch determinism", flush=True)


try:
    _pin_triton_autotune()
except Exception as exc:  # degrade loudly, never break the interpreter
    print(f"[pin_autotune] FAILED to pin triton autotune: {exc!r}", flush=True)
