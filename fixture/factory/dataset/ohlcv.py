from datetime import date

import polars as pl

from dataset.schema import OHLCV


def factory_ohlcv() -> OHLCV:
    return pl.DataFrame(
        {
            "Date": [date(2000, 1, d) for d in range(1, 5)],
            "Open": [100.0, 200.0, 300.0, 100.0],
            "High": [105.0, 210.0, 345.0, 120.0],
            "Low": [95.0, 190.0, 255.0, 80.0],
            "Close": [101.0, 204.0, 309.0, 100.0],
            "Volume": [1000, 1100, 1200, 1000],
        }
    )
