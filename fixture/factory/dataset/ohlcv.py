from datetime import date

import polars as pl

from dataset.schema import OHLCV


def factory_ohlcv() -> OHLCV:
    """
    Open:   100ずつ増える。ただし最後は101.0でCloseと同じ値
    High:   Openを基準に5%ずつ日毎に上昇幅が上昇
    Low:    penを基準に5%ずつ日毎に下落幅が下落
    Close:  Openを基準に日毎に1%ずつ上昇幅が上昇。ただし最後はOpenと同じ価格となる
    Volume: 1000から日毎に100ずつ上昇。ただし最終日は初めと同じ値
    """
    return pl.DataFrame(
        {
            "Date": [date(2000, 1, d) for d in range(1, 5)],
            "Open": [100.0, 200.0, 300.0, 101.0],
            "High": [105.0, 210.0, 345.0, 120.0],
            "Low": [95.0, 190.0, 255.0, 80.0],
            "Close": [101.0, 204.0, 309.0, 101.0],
            "Volume": [1000, 1100, 1200, 1000],
        }
    )
