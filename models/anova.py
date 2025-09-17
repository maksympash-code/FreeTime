"""
One-Way ANOVA without SciPy + permutation p-value.
Default factor: 'marketing_channel' vs metric 'total_value' from transactions.csv
"""
from typing import Dict, List
from .io_utils import load_csv
from collections import defaultdict
import random
import math


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def group_metric(records, factor_col="marketing_channel", metric_col="total_value"):
    groups = defaultdict(list)
    for r in records:
        y = _to_float(r.get(metric_col, "nan"))
        g = r.get(factor_col, "NA")
        if not math.isnan(y):
            groups[g].append(y)
    return groups


def anova_oneway(groups: Dict[str, List[float]]):
    all_values = [v for arr in groups.values() for v in arr]
    N = len(all_values)
    k = len(groups)
    grand_mean = sum(all_values) / N

    ss_between = sum(len(arr) * (sum(arr) / len(arr) - grand_mean) ** 2 for arr in groups.values())
    ss_within = sum(sum((v - sum(arr) / len(arr)) ** 2 for v in arr) for arr in groups.values())

    df_between = k - 1
    df_within = N - k

    ms_between = ss_between / df_between if df_between > 0 else float('nan')
    ms_within = ss_within / df_within if df_within > 0 else float('nan')

    F = ms_between / ms_within if ms_within > 0 else float('inf')
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else float('nan')

    return {
        "k_groups": k,
        "N": N,
        "F": F,
        "df_between": df_between,
        "df_within": df_within,
        "eta_squared": eta_sq,
    }


def permutation_pvalue(groups: Dict[str, List[float]], iters: int = 5000, seed: int = 42):
    rnd = random.Random(seed)
    values = [v for arr in groups.values() for v in arr]
    observed = anova_oneway(groups)["F"]

    sizes = {g: len(arr) for g, arr in groups.items()}
    keys = list(sizes.keys())
    n_sizes = [sizes[k] for k in keys]

    count = 0
    for _ in range(iters):
        rnd.shuffle(values)
        start = 0
        perm_groups = {}
        for i, g in enumerate(keys):
            s = n_sizes[i]
            perm_groups[g] = values[start:start + s]
            start += s
        Fp = anova_oneway(perm_groups)["F"]
        if Fp >= observed - 1e-12:
            count += 1
    pval = (count + 1) / (iters + 1)
    return pval


def run_anova(transactions_csv: str, factor="marketing_channel", metric="total_value", iters=5000):
    records, _ = load_csv(transactions_csv)
    groups = group_metric(records, factor, metric)
    stats = anova_oneway(groups)
    p_perm = permutation_pvalue(groups, iters=iters, seed=123)
    stats["p_value_perm"] = p_perm
    return stats, {g: (len(arr), sum(arr) / len(arr)) for g, arr in groups.items()}