from datetime import date

import polars as pl

from dataset.schema import OHCLV


def factory_ohclv() -> OHCLV:
    return pl.DataFrame(
        {
            "Date": [date(2000, 1, d) for d in range(1, 5)],
            "Open": [100.0, 200.0, 300.0, 400.0],
            "High": [110.0, 220.0, 330.0, 440.0],
            "Low": [90.0, 180.0, 270.0, 360.0],
            "Close": [105.0, 210.0, 315.0, 420.0],
            "Volume": [1000, 1100, 1200, 1300],
        }
    )
