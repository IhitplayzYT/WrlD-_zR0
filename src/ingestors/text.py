import polars as pl
def ingest_txt(path: str):
    ret = None
    with open(path) as f:
        ret = pl.DataFrame({"price":  list(map(lambda x: float(x.strip()), f.read().splitlines()))})
    return ret
