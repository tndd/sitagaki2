from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime

from domain.dataset.ohlcv import Ohlcv


def factory_ohlcv() -> Ohlcv:
    df = DataFrame(
        {
            "Date": to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2021-01-07",
                    "2021-01-08",
                    "2021-01-09",
                ]
            ),
            "Open": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
            "High": [110.0, 210.0, 310.0, 410.0, 510.0, 610.0, 710.0, 810.0, 910.0],
            "Low": [90.0, 190.0, 290.0, 390.0, 490.0, 590.0, 690.0, 790.0, 890.0],
            "Close": [105.0, 205.0, 305.0, 405.0, 505.0, 605.0, 705.0, 805.0, 905.0],
            "Volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        }
    )
    return Ohlcv(df)


def factory_ohlcv_random_walk(n: int = 1000) -> Ohlcv:
    """
    デフォルトで1000件のデータを生成する(nで設定)

    OHLCVデータとして辻褄が合うように以下の条件を満たす:
    - High >= Open, Close, Low
    - Low <= Open, Close, High
    - 日付は連続する (取引日として週末を除く)
    - 価格変動は現実的な範囲内
    """
    # 初期価格と日付を設定
    base_price = 100.0
    start_date = datetime(2020, 1, 1)

    # 日付配列を生成（週末を除く営業日1000日分）
    dates = []
    current_date = start_date
    while len(dates) < n:
        # 土日を除外 (5: 土曜, 6: 日曜)
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)

    # 価格変動のシミュレーション
    np.random.seed(42)  # 再現性のためにシード値を設定

    # ランダムウォークで価格を生成
    returns = np.random.normal(
        0.0005, 0.015, n
    )  # 平均リターン0.05%、ボラティリティ1.5%
    price_multipliers = np.exp(np.cumsum(returns))
    base_prices = base_price * price_multipliers

    # データフレームを作成
    data = []
    for i, date in enumerate(dates):
        # その日の基本価格
        price = base_prices[i]

        # その日のボラティリティ（価格帯の幅）
        volatility = price * np.random.uniform(0.01, 0.03)  # 1〜3%のボラティリティ

        # 始値・終値の生成
        open_price = price * np.random.uniform(0.99, 1.01)
        close_price = price * np.random.uniform(0.99, 1.01)

        # 高値・安値の生成（辻褄が合うように）
        high_price = max(open_price, close_price) + volatility * np.random.uniform(0, 1)
        low_price = min(open_price, close_price) - volatility * np.random.uniform(0, 1)

        # 出来高の生成（価格変動に応じて増減するように）
        price_change_ratio = abs((close_price / open_price) - 1)
        volume = int(np.random.normal(1000000, 500000) * (1 + price_change_ratio * 10))
        volume = max(100000, volume)  # 最低出来高を設定

        data.append(
            {
                "Date": date,
                "Open": round(open_price, 2),
                "High": round(high_price, 2),
                "Low": round(low_price, 2),
                "Close": round(close_price, 2),
                "Volume": volume,
            }
        )

    # DataFrameを作成
    df = pd.DataFrame(data)

    return Ohlcv(df)
