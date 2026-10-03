import polars as pl

def ingest_csv(path:str):
    df = pl.read_csv(path)
    if "price" not in df.columns:
        if len(df.columns) > 0:
            df = df.rename({df.columns[0]: "price"})
        else:
            raise ValueError("CSV must have at least one column to use as 'price'")
    return df

def ingest_xlsx(path:str):
    df = pl.read_excel(path,encoding="utf-8")
    if "price" not in df.columns:
        if len(df.columns) > 0:
            df = df.rename({df.columns[0]: "price"})
        else:
            raise ValueError("Excel must have at least one column to use as 'price'")
    return df

def ingest_excel(path:str):
    return ingest_xlsx(path) if path.endswith("xls") or path.endswith("xlsx") else ingest_csv(path)



    