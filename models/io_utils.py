def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def load_csv(path):
    """
    Returns (records:list[dict], columns:list[str]).
    Uses pandas if available; falls back to csv module.
    """
    pd = try_import_pandas()
    if pd is not None:
        df = pd.read_csv(path)
        return df.to_dict(orient="records"), list(df.columns)
    else:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows, reader.fieldnames or []


def write_csv(path, rows, fieldnames):
    pd = try_import_pandas()
    if pd is not None:
        import pandas as pd
        df = pd.DataFrame(rows, columns=fieldnames)
        df.to_csv(path, index=False)
    else:
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)