from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from domain.dataset.ohlcv.schema import Ohlcv


def factory_ohlcv() -> Ohlcv:
    """
    Open:   100ずつ増える。ただし最後は101.0でCloseと同じ値
    High:   Openを基準に5%ずつ日毎に上昇幅が上昇
    Low:    Openを基準に5%ずつ日毎に下落幅が下落
    Close:  Openを基準に日毎に1%ずつ上昇幅が上昇。ただし最後はOpenと同じ価格となる
    Volume: 1000から日毎に100ずつ上昇。ただし最終日は初めと同じ値
    """
    return Ohlcv(
        pl.DataFrame(
            {
                "Date": [datetime(2000, 1, d) for d in range(1, 5)],
                "Open": [100.0, 200.0, 300.0, 101.0],
                "High": [105.0, 210.0, 345.0, 120.0],
                "Low": [95.0, 190.0, 255.0, 80.0],
                "Close": [101.0, 204.0, 309.0, 101.0],
                "Volume": [1000, 1100, 1200, 1000],
            }
        )
    )


def factory_ohlcv_1000() -> Ohlcv:
    """
    固定の1000件のOHLCVデータを生成する

    > head()
    ┌─────────────────────┬─────────┬─────────┬─────────┬─────────┬────────┐
    │ Date                ┆ Open    ┆ High    ┆ Low     ┆ Close   ┆ Volume │
    │ ---                 ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---    │
    │ datetime[ns]        ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ i64    │
    ╞═════════════════════╪═════════╪═════════╪═════════╪═════════╪════════╡
    │ 2023-01-01 00:00:00 ┆ 137.454 ┆ 139.305 ┆ 134.837 ┆ 139.181 ┆ 5713   │
    │ 2023-01-02 00:00:00 ┆ 195.071 ┆ 200.49  ┆ 192.601 ┆ 198.038 ┆ 4092   │
    │ 2023-01-03 00:00:00 ┆ 173.199 ┆ 181.928 ┆ 164.136 ┆ 170.704 ┆ 5159   │
    │ 2023-01-04 00:00:00 ┆ 159.866 ┆ 167.188 ┆ 157.371 ┆ 161.115 ┆ 7308   │
    │ 2023-01-05 00:00:00 ┆ 115.602 ┆ 123.668 ┆ 112.883 ┆ 116.319 ┆ 8478   │
    └─────────────────────┴─────────┴─────────┴─────────┴─────────┴────────┘
    """
    N_DECIMAL = 3
    # 再現性のためにシードを固定
    np.random.seed(42)
    # ohlcvの部品
    dates = pd.date_range(
        start="2023-01-01",
        periods=1000,
        freq="D",
    ).astype("datetime64[us]")  # pd.date_rangeはデフォルトではnsに変換してしまう
    open_prices = np.round(np.random.uniform(100, 200, size=1000), N_DECIMAL)
    high_prices = np.round(open_prices + np.random.uniform(0, 10, size=1000), N_DECIMAL)
    low_prices = np.round(open_prices - np.random.uniform(0, 10, size=1000), N_DECIMAL)
    close_prices = np.round(
        open_prices + np.random.uniform(-5, 5, size=1000), N_DECIMAL
    )
    volumes = np.random.randint(1000, 10000, size=1000)

    return Ohlcv(
        pl.DataFrame(
            {
                "Date": dates,
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volumes,
            }
        )
    )


def factory_ohlcv_cycle(start_date="2000-01-01", length=100) -> Ohlcv:
    # 基本の価格パターン（階段状に上昇）
    base_prices = np.concatenate(
        [np.linspace(100, 105, 30), np.linspace(105, 95, 40), np.linspace(95, 110, 30)]
    )
    dates = [
        datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)
        for i in range(length)
    ]
    ohlc_data = []
    prev_close = 100.0
    for i in range(length):
        # 5日周期でパターン変化
        cycle = i % 5
        if cycle == 0:
            open_price = prev_close * 1.01
        elif cycle == 3:
            open_price = prev_close * 0.99
        else:
            open_price = prev_close
        close_price = base_prices[i] + np.random.normal(0, 0.5)
        high = max(open_price, close_price) + abs(np.random.normal(0.2, 0.1))
        low = min(open_price, close_price) - abs(np.random.normal(0.2, 0.1))
        volume = int(np.random.uniform(10000, 50000))
        ohlc_data.append(
            {
                "Date": dates[i],
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close_price, 2),
                "Volume": volume,
            }
        )
        prev_close = close_price
    return Ohlcv(pl.DataFrame(ohlc_data))


def factory_ohlcv_brown(num_rows: int = 10_0000) -> Ohlcv:
    """ランダムなOHLCVデータを生成する関数"""

    # ベース日時の生成（1分足を想定）
    start_date = pl.datetime(2000, 1, 1)  # ←引数を個別に指定
    base_date = pl.datetime_range(
        start=start_date,
        end=start_date + pl.duration(minutes=num_rows - 1),
        interval="1m",
        eager=True,
    ).alias("Date")

    # ランダムな価格変動（幾何ブラウン運動を模倣）
    returns = np.random.normal(0, 0.0001, num_rows)
    close_prices = 100.0 * np.exp(np.cumsum(returns))

    return Ohlcv(
        pl.DataFrame(
            {
                "Date": base_date,
                "Open": close_prices * np.random.uniform(0.99, 1.01, num_rows),
                "High": close_prices * np.random.uniform(1.0, 1.02, num_rows),
                "Low": close_prices * np.random.uniform(0.98, 1.0, num_rows),
                "Close": close_prices,
                "Volume": np.random.normal(1_000_000, 100_000, num_rows).astype(
                    np.int64
                ),
            }
        ).cast(Ohlcv.SCHEMA)
    )


if __name__ == "__main__":
    ohlcv = factory_ohlcv()
    print(ohlcv.df)
