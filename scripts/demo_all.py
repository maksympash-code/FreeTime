"""
Run all models on the synthetic data and print key outputs.
Run: python scripts/demo_all.py
"""
from pathlib import Path
from models.market_basket import run_market_basket
from models.anova import run_anova
from models.regression import run_regression
from models.ab_test import run_abtest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    tx_csv = str(DATA / "transactions.csv")
    ab_csv = str(DATA / "ab_sessions.csv")

    print("\n=== Market Basket (Top rules) ===")
    rules = run_market_basket(tx_csv, min_support=0.05, min_confidence=0.35, top_k=10)
    for r in rules:
        print(f"{r['antecedent']} -> {r['consequent']} | support={r['support']}, conf={r['confidence']}, lift={r['lift']}")

    print("\n=== One-Way ANOVA (total_value ~ marketing_channel) ===")
    stats, means = run_anova(tx_csv, factor="marketing_channel", metric="total_value", iters=3000)
    print("group means (n, mean):", {k: (v[0], round(v[1], 2)) for k, v in means.items()})
    print(f"F({stats['df_between']},{stats['df_within']}) = {stats['F']:.3f}, eta^2={stats['eta_squared']:.3f}, p_perm≈{stats['p_value_perm']:.4f}")

    print("\n=== Regression (OLS) ===")
    beta, metrics = run_regression(tx_csv)
    print("β coefficients (order):", metrics["feature_order"])
    print([round(b, 4) for b in beta])
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if k.startswith(("rmse", "r2", "n_"))})

    print("\n=== A/B test (two-proportion z-test) ===")
    res = run_abtest(ab_csv)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["variant_A"].items()})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["variant_B"].items()})
    print(f"z={res['z_stat']:.3f}, p={res['p_value']:.4f}, uplift={res['relative_uplift']:.3%}")


if __name__ == "__main__":
    main()