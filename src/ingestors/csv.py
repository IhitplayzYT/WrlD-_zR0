import polars as pl

def ingest_csv(path:str):
    """ CSV Ingestor """
    return pl.read_csv(path)

def ingest_xlsx(path:str):
    """ Excel Ingestor """
    return pl.read_Excel(path,encoding="utf-8")

def ingest_excel(path:str):
    """ Exposed Excel/CSV Ingestor"""
    return ingest_xlsx(path) if path.endswith("xls") or path.endswith("xlsx") else ingest_csv(path)



    