"""
Generate synthetic retail dataset (deterministic with seed).
Run: python scripts/generate_data.py
"""
import random
from pathlib import Path
from datetime import datetime, timedelta
from models.io_utils import write_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rng():
    return random.Random(20250914)


def build_catalog(r):
    categories = {
        "Dairy": ["Milk", "Cheese", "Yogurt", "Butter"],
        "Bakery": ["Bread", "Croissant", "Bagel", "Muffin"],
        "Produce": ["Apple", "Banana", "Tomato", "Lettuce", "Onion", "Potato"],
        "Snacks": ["Chips", "Chocolate", "Cookies", "Nuts"],
        "Beverages": ["Soda", "Juice", "Water", "Coffee", "Tea", "Wine"],
        "Cereal": ["Corn Flakes", "Oatmeal", "Muesli"],
        "Meat": ["Chicken", "Beef", "Sausage"],
    }
    products = []
    pid = 1001
    for cat, names in categories.items():
        for name in names:
            base = {
                "Dairy": (1.1, 2.5),
                "Bakery": (0.7, 2.0),
                "Produce": (0.5, 1.5),
                "Snacks": (1.0, 3.0),
                "Beverages": (0.6, 8.0),
                "Cereal": (1.5, 4.0),
                "Meat": (3.0, 9.0),
            }[cat]
            price = round(base[0] + (base[1] - base[0]) * r.random(), 2)
            products.append({"product_id": str(pid), "name": name, "category": cat, "price": price})
            pid += 1
    return products


def build_customers(r, n=300):
    segs = ["retail", "loyalty", "online"]
    return [{"customer_id": str(cid), "segment": r.choices(segs, weights=[0.5, 0.3, 0.2], k=1)[0]} for cid in range(1, n + 1)]


def sample_basket(r, products):
    size = r.choices([1, 2, 3, 4, 5, 6], weights=[0.05, 0.25, 0.3, 0.25, 0.1, 0.05], k=1)[0]
    picks = set()
    cats = list(set([p["category"] for p in products]))
    chosen_cat = r.choice(cats)
    cat_pool = [p for p in products if p["category"] == chosen_cat]
    while len(picks) < size:
        p = r.choice(cat_pool if r.random() < 0.6 else products)
        picks.add(p["product_id"])

    id_by_name = {p["name"]: p["product_id"] for p in products}

    def maybe_add(a, b, prob):
        if id_by_name.get(a) in picks and r.random() < prob:
            picks.add(id_by_name[b])

    maybe_add("Bread", "Butter", 0.35)
    maybe_add("Milk", "Corn Flakes", 0.40)
    maybe_add("Milk", "Oatmeal", 0.25)
    maybe_add("Cheese", "Wine", 0.30)
    maybe_add("Chips", "Soda", 0.45)
    maybe_add("Coffee", "Cookies", 0.25)
    return list(picks)


def build_transactions(r, products, customers, days=60):
    start = datetime(2025, 6, 1)
    transactions = []
    marketing_channels = ["none", "email", "social"]
    prices = {p["product_id"]: float(p["price"]) for p in products}
    tid = 1
    for d in range(days):
        date = start + timedelta(days=d)
        n_tx = r.randint(30, 70)
        for _ in range(n_tx):
            customer = r.choice(customers)
            items = sample_basket(r, products)
            has_promo = r.random() < 0.35
            discount = r.uniform(0.05, 0.20) if has_promo else 0.0
            total = sum(prices[i] for i in items)
            total_after = round(total * (1.0 - discount), 2)
            mkt = r.choices(marketing_channels, weights=[0.55, 0.25, 0.20], k=1)[0]
            transactions.append({
                "transaction_id": str(tid),
                "date": date.strftime("%Y-%m-%d"),
                "customer_id": customer["customer_id"],
                "items": ";".join(items),
                "items_count": len(items),
                "total_value": total_after,
                "marketing_channel": mkt,
                "has_promo": int(has_promo),
            })
            tid += 1
    return transactions


def build_inventory(r, products, days=60):
    start = datetime(2025, 6, 1)
    inv = []
    stock = {p["product_id"]: r.randint(20, 120) for p in products}
    for d in range(days):
        date = start + timedelta(days=d)
        for p in products:
            pid = p["product_id"]
            sold = max(0, int(r.gauss(mu=3, sigma=2)))
            stock[pid] = max(0, stock[pid] - sold)
            if stock[pid] < 10 or (r.random() < 0.05):
                add = r.randint(15, 80)
                stock[pid] += add
                action = "replenish"; qty = add
            else:
                action = "hold"; qty = 0
            inv.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "stock_level": stock[pid],
                "action": action,
                "qty": qty
            })
    return inv


def build_ab_sessions(r, n=4000):
    rows = []
    for i in range(1, n + 1):
        variant = "A" if r.random() < 0.5 else "B"
        pA, pB = 0.10, 0.125
        p = pA if variant == "A" else pB
        converted = 1 if r.random() < p else 0
        rows.append({"session_id": i, "variant": variant, "converted": converted})
    return rows


def main():
    r = rng()
    DATA.mkdir(parents=True, exist_ok=True)
    products = build_catalog(r)
    customers = build_customers(r, n=300)
    transactions = build_transactions(r, products, customers, days=60)
    inventory = build_inventory(r, products, days=60)
    ab = build_ab_sessions(r, n=4000)

    write_csv(DATA / "products.csv", products, ["product_id", "name", "category", "price"])
    write_csv(DATA / "customers.csv", customers, ["customer_id", "segment"])
    write_csv(DATA / "transactions.csv", transactions, ["transaction_id", "date", "customer_id", "items", "items_count", "total_value", "marketing_channel", "has_promo"])
    write_csv(DATA / "inventory_log.csv", inventory, ["date", "product_id", "stock_level", "action", "qty"])
    write_csv(DATA / "ab_sessions.csv", ab, ["session_id", "variant", "converted"])

    print("Data generated into ./data")


if __name__ == "__main__":
    main()