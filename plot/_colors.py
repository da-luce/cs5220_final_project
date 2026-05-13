"""Shared per-variant colors so the same variant looks identical across plots."""

VARIANT_COLORS = {
    "tiny":    "tab:blue",
    "classic": "tab:green",
    "quick":   "tab:purple",
    "barrage": "tab:orange",
}


def color_for(variant, fallback_cycle):
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    return next(fallback_cycle)
