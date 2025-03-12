from datetime import datetime, timedelta

import numpy as np
import polars as pl

from dataset.ohlcv.schema import OHLCV


def factory_ohlcv() -> OHLCV:
    """
    Open:   100ずつ増える。ただし最後は101.0でCloseと同じ値
    High:   Openを基準に5%ずつ日毎に上昇幅が上昇
    Low:    Openを基準に5%ずつ日毎に下落幅が下落
    Close:  Openを基準に日毎に1%ずつ上昇幅が上昇。ただし最後はOpenと同じ価格となる
    Volume: 1000から日毎に100ずつ上昇。ただし最終日は初めと同じ値
    """
    return OHLCV(
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


def factory_ohlcv_cycle(start_date="2000-01-01", length=100) -> OHLCV:
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
    return OHLCV(df=pl.DataFrame(ohlc_data))


def factory_ohlcv_brown(num_rows: int = 10_0000) -> OHLCV:
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

    return OHLCV(
        df=pl.DataFrame(
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
        ).cast(OHLCV.schema)
    )


if __name__ == "__main__":
    ohlcv = factory_ohlcv()
    print(ohlcv.df)
