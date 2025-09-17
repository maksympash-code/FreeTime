"""
Simple Apriori for Market Basket Analysis (offline, pure-Python).
Input: transactions.csv with "items" column (semicolon-separated product_ids).
"""
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, FrozenSet
from .io_utils import load_csv


def parse_transactions(path: str) -> List[FrozenSet[str]]:
    records, _ = load_csv(path)
    tx = []
    for r in records:
        items = [x.strip() for x in str(r.get("items", "")).split(";") if x.strip()]
        tx.append(frozenset(items))
    return tx


def apriori(transactions: List[FrozenSet[str]], min_support: float = 0.05):
    n = len(transactions)
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] += 1
    L = {iset: c / n for iset, c in item_counts.items() if c / n >= min_support}
    all_freq = dict(L)

    k = 2
    current_L = set(L.keys())
    while current_L:
        candidates = set()
        L_list = list(current_L)
        for i in range(len(L_list)):
            for j in range(i + 1, len(L_list)):
                a = L_list[i]
                b = L_list[j]
                union = a | b
                if len(union) == k:
                    if all((frozenset(s) in current_L) for s in combinations(union, k - 1)):
                        candidates.add(union)

        counts = defaultdict(int)
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    counts[c] += 1

        Lk = {iset: cnt / n for iset, cnt in counts.items() if cnt / n >= min_support}
        all_freq.update(Lk)
        current_L = set(Lk.keys())
        k += 1

    return all_freq


def generate_rules(frequent_itemsets: Dict[FrozenSet[str], float], min_confidence: float = 0.3):
    rules = []
    supp = frequent_itemsets
    for itemset, support_X in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for A_tuple in combinations(items, r):
                A = frozenset(A_tuple)
                B = itemset - A
                sA = supp.get(A, 0.0)
                sB = supp.get(B, 0.0)
                if sA == 0:
                    continue
                conf = support_X / sA
                if conf >= min_confidence and sB > 0:
                    lift = conf / sB
                    leverage = support_X - sA * sB
                    rules.append({
                        "antecedent": tuple(sorted(A)),
                        "consequent": tuple(sorted(B)),
                        "support": round(support_X, 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 4),
                        "leverage": round(leverage, 4),
                    })
    rules.sort(key=lambda d: (d["lift"], d["confidence"]), reverse=True)
    return rules


def run_market_basket(transactions_csv: str, min_support=0.05, min_confidence=0.35, top_k=10):
    tx = parse_transactions(transactions_csv)
    freq = apriori(tx, min_support=min_support)
    rules = generate_rules(freq, min_confidence=min_confidence)
    return rules[:top_k]