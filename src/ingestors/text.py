import polars as pl

def ingest_txt(path: str):
    """ Ingest Plain Text data for the Stock Data - returns DataFrame with 'price' column """
    ret = None
    with open(path) as f:
        prices = list(map(lambda x: float(x.strip()), f.read().splitlines()))
        ret = pl.DataFrame({"price": prices})
        f.close()
    return ret
