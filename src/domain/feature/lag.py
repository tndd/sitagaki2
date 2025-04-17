from numpy import log
from pandas import DataFrame

from domain.dataset.ohlcv import Ohlcv
from domain.feature.common import OhlcvFeature


class LagCloses10(OhlcvFeature):
    """
    10日分の終値の変化率の特徴量
    """

    SCHEMA = {
        "Date": "INDEX",
        "l0": float,
        "l1": float,
        "l2": float,
        "l3": float,
        "l4": float,
        "l5": float,
        "l6": float,
        "l7": float,
        "l8": float,
        "l9": float,
        "l10": float,
    }

    def __init__(self, ohlcv: Ohlcv) -> None:
        super().__init__(
            ohlcv,
            label="l0",
        )

    @staticmethod
    def _feature_df_source(ohlcv: Ohlcv) -> DataFrame:
        # n=10 を指定して汎用関数を呼び出す
        return derive_lag_df_from_ohlcv(ohlcv, n=10)


def derive_lag_df_from_ohlcv(ohlcv: Ohlcv, n: int) -> DataFrame:
    """
    n日分の終値の対数差分の特徴量を生成し、特徴量のみのDataFrameを返す。
    インデックスは元のOHLCVデータから引き継ぎ、欠損値を含む行は削除する。

    nが0以下の場合:
        空のDataFrameを返す仕様とする。
    """
    # ohlcvとインデックスを同じくする新たな特徴量のDataFrameを作成
    feature_df = DataFrame(index=ohlcv.df.index)
    # nが0以下の場合は空のDataFrameを返す
    if n <= 0:
        return feature_df
    # 対数差分の計算
    close = ohlcv.df["Close"]
    for i in range(n + 1):
        shifted_close = close.shift(i)
        next_shifted_close = close.shift(i + 1)
        # 安全に計算できる行を計算
        mask = (
            shifted_close.notna()
            & next_shifted_close.notna()
            & (next_shifted_close != 0)
        )
        feature_df[f"l{i}"] = float("nan")  # NoneだとObject型と判定されエラー
        # 計算できない部分はスキップされnanのまま
        feature_df.loc[mask, f"l{i}"] = log(
            shifted_close[mask] / next_shifted_close[mask]
        )
    # 全てのlagが満たされていない行は除外する
    return feature_df.dropna()
