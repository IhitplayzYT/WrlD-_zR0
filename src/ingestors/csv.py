import polars as pl

def ingest_csv(path:str):
    """ CSV Ingestor - returns DataFrame with 'price' column """
    df = pl.read_csv(path)
    # If 'price' column doesn't exist, try to find a similar column or rename first column
    if "price" not in df.columns:
        if len(df.columns) > 0:
            df = df.rename({df.columns[0]: "price"})
        else:
            raise ValueError("CSV must have at least one column to use as 'price'")
    return df

def ingest_xlsx(path:str):
    """ Excel Ingestor - returns DataFrame with 'price' column """
    df = pl.read_excel(path,encoding="utf-8")
    # If 'price' column doesn't exist, try to find a similar column or rename first column
    if "price" not in df.columns:
        if len(df.columns) > 0:
            df = df.rename({df.columns[0]: "price"})
        else:
            raise ValueError("Excel must have at least one column to use as 'price'")
    return df

def ingest_excel(path:str):
    """ Exposed Excel/CSV Ingestor - returns DataFrame with 'price' column"""
    return ingest_xlsx(path) if path.endswith("xls") or path.endswith("xlsx") else ingest_csv(path)



    