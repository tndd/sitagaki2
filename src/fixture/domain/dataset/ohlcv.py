from datetime import datetime, timedelta

import numpy as np
import polars as pl
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
    注意: 高速化のためPolarsを使用している

    OHLCVデータとして辻褄が合うように以下の条件を満たす:
    - High >= Open, Close, Low
    - Low <= Open, Close, High
    - 日付は1分単位で連続する
    - 価格変動は現実的な範囲内
    """
    # シード値を設定
    np.random.seed(42)

    # 初期価格設定
    base_price = 100.0

    # 日付配列を生成（NumPyを使用して生成し、後でPolarsに変換）
    start_date = datetime(2020, 1, 1)
    # 分単位のオフセットを生成して日付配列を作成
    date_offsets = np.arange(n)
    dates = np.array(
        [start_date + timedelta(minutes=int(offset)) for offset in date_offsets],
        dtype="datetime64[ns]",
    )

    # ランダムウォークを生成
    returns = np.random.normal(loc=0, scale=0.0001, size=n)
    price_multipliers = np.cumprod(1 + returns)
    base_prices = base_price * price_multipliers

    # ボラティリティ（固定の小さなボラティリティ）
    volatility = base_prices * 0.001

    # ランダム変動を生成
    open_rand = np.random.uniform(0.9995, 1.0005, n)
    close_rand = np.random.uniform(0.9995, 1.0005, n)
    high_rand = np.random.uniform(0, 1, n)
    low_rand = np.random.uniform(0, 1, n)

    # 価格データを計算
    open_prices = base_prices * open_rand
    close_prices = base_prices * close_rand

    # 高値と安値を計算（辻褄が合うように）
    max_oc = np.maximum(open_prices, close_prices)
    min_oc = np.minimum(open_prices, close_prices)
    high_prices = max_oc + volatility * high_rand
    low_prices = min_oc - volatility * low_rand

    # 出来高を計算
    price_change_ratio = np.abs((close_prices / open_prices) - 1)
    volume_base = np.random.normal(10000, 5000, n)
    volumes = np.maximum(
        1000, (volume_base * (1 + price_change_ratio * 10)).astype(int)
    )

    # Polarsデータフレームを作成
    pl_df = pl.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_prices, 2),
            "High": np.round(high_prices, 2),
            "Low": np.round(low_prices, 2),
            "Close": np.round(close_prices, 2),
            "Volume": volumes,
        }
    )

    # PandasのDataFrameに変換してOhlcvオブジェクトとして返す
    pd_df = pl_df.to_pandas()
    return Ohlcv(pd_df)
