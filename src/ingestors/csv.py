import numpy as np
import polars as pl


def normalize_prices(df):
    columns = {c.lower().strip().replace('_', ' '): c for c in df.columns}
    price = next((columns[c] for c in ('price', 'adj close', 'adjusted close', 'close') if c in columns), None)
    if price is None:
        raise ValueError('Data needs an explicit price, adjusted close, or close column')
    if price != 'price':
        df = df.rename({price: 'price'})
    dates = next((columns[c] for c in ('date', 'datetime', 'timestamp') if c in columns), None)
    if dates:
        if df[dates].dtype == pl.String:
            df = df.with_columns(pl.col(dates).str.to_datetime(strict=True))
        df = df.sort(dates)
        if df[dates].n_unique() != len(df):
            raise ValueError('Duplicate market timestamps')
    df = df.with_columns(pl.col('price').cast(pl.Float64))
    values = np.asarray(df['price'].to_list(), dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Prices must be finite and positive; missing values need explicit cleaning')
    return df


def ingest_csv(path):
    return normalize_prices(pl.read_csv(path))


def ingest_xlsx(path):
    return normalize_prices(pl.read_excel(path))


def ingest_excel(path):
    return ingest_xlsx(path) if path.lower().endswith(('.xls', '.xlsx')) else ingest_csv(path)
