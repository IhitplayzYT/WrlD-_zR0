import polars as pl

def ingest_txt(path: str):
    """ Ingest Plain Text data for the Stock Data """
    ret = None
    with open(path) as f:
        ret = pl.DataFrame(map(lambda x: float(x.trim()),f.read().splitlines()))
        f.close()
    return ret
