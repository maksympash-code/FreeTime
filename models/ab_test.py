"""
Two-proportion z-test for conversion rates (A/B test). No SciPy.
"""
from .io_utils import load_csv
import math


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def two_proportion_z_test(success_a, total_a, success_b, total_b, alternative="two-sided"):
    p_pool = (success_a + success_b) / (total_a + total_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))
    if se == 0:
        return float('nan'), float('nan')
    z = ((success_b / total_b) - (success_a / total_a)) / se

    if alternative == "two-sided":
        p = 2 * (1 - _norm_cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - _norm_cdf(z)
    else:
        p = _norm_cdf(z)
    return z, p


def run_abtest(csv_path: str, variant_col="variant", conv_col="converted"):
    rows, _ = load_csv(csv_path)
    sa = ta = sb = tb = 0
    for r in rows:
        v = str(r.get(variant_col, "A")).strip().upper()
        c = 1 if str(r.get(conv_col, "0")).strip() in ("1", "True", "true", "yes") else 0
        if v == "A":
            ta += 1; sa += c
        else:
            tb += 1; sb += c

    z, p = two_proportion_z_test(sa, ta, sb, tb, alternative="two-sided")
    cr_a = sa / ta if ta else float('nan')
    cr_b = sb / tb if tb else float('nan')
    uplift = (cr_b - cr_a) / cr_a if cr_a > 0 else float('nan')
    return {
        "variant_A": {"n": ta, "conversions": sa, "cr": cr_a},
        "variant_B": {"n": tb, "conversions": sb, "cr": cr_b},
        "z_stat": z, "p_value": p, "relative_uplift": uplift
    }