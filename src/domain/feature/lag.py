from numpy import log
from pandas import DataFrame

from domain.dataset.ohlcv import Ohlcv
from domain.feature.common import OhlcvFeature

LAG_CLOSES_10_DEFINITION = {
    "l0": float,  # Label
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
# LAG特徴量全体のラベルカラムはl0という前提
LAG_CLOSES_LABEL = "l0"


class LagCloses10(OhlcvFeature):
    """
    10日分の終値の変化率の特徴量
    """

    def __init__(self, ohlcv: Ohlcv) -> None:
        super().__init__(
            df=derive_lag_df_from_ohlcv(ohlcv, n=10),
            definition=LAG_CLOSES_10_DEFINITION,
            label=LAG_CLOSES_LABEL,
        )


def derive_lag_df_from_ohlcv(ohlcv: Ohlcv, n: int) -> DataFrame:
    """
    n日分の終値の対数差分の特徴量を生成し、元のOHLCVカラムを残したDataFrameを返す。
    インデックスは元のOHLCVデータから引き継ぎ、欠損値を含む行は削除する。

    nが0以下の場合の挙動:
        ohlcv.df をそのまま返す。
    """
    if n <= 0:
        # nが0以下なら、元のDataFrameをそのまま返す
        return ohlcv.df
    # nが1以上なら、雛形を作って返す
    columns_to_add = {f"l{i}": float("nan") for i in range(n + 1)}
    feature_df = DataFrame(columns_to_add, index=ohlcv.df.index)
    # 対数差分の計算
    close = ohlcv.df["Close"]
    # 計算できない部分はスキップされnanのまま
    for i in range(n + 1):
        shifted_close = close.shift(i)
        next_shifted_close = close.shift(i + 1)
        # 安全に計算できる行を計算
        mask = (
            shifted_close.notna()
            & next_shifted_close.notna()
            & (next_shifted_close != 0)
        )
        feature_df.loc[mask, f"l{i}"] = log(
            shifted_close[mask] / next_shifted_close[mask]
        )
    # 特徴量DataFrameと元のOHLCV DataFrameを結合
    # how="inner" を使用して、両方のDataFrameに存在するインデックスのみを保持する
    merged_df = ohlcv.df.join(feature_df, how="inner")
    # 全てのlagが満たされていない行は除外する
    return merged_df.dropna()
