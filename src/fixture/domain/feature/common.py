from pandas import DataFrame

from domain.feature.common import OhlcvFeature

OHLCV_FEATURE_IMPL_DEFINITION = {
    "OF0": float,  # Label
    "OF1": float,
    "OF2": float,
    "OF3": float,
    "OF4": float,
    "OF5": float,
    "OF6": float,
}


class OhlcvFeatureImpl(OhlcvFeature):
    def __init__(self, df: DataFrame) -> None:
        # インデックス指定なし
        super().__init__(
            df=df,
            definition=OHLCV_FEATURE_IMPL_DEFINITION,
            label="OF0",
        )


def factory_ohlcv_feature_impl() -> OhlcvFeatureImpl:
    """
    OhlcvFeatureImplを作成する。
    各カラムの値が'カラム番号.行番号'の形式となる。

    # TODO: ohlcvとの結合がうまくいっていない
    """
    data = {
        col: [
            f"{list(OHLCV_FEATURE_IMPL_DEFINITION.keys()).index(col)}.{i * 0.1}"
            for i in range(100)
        ]
        for col in OHLCV_FEATURE_IMPL_DEFINITION.keys()
    }
    df = DataFrame(data)
    return OhlcvFeatureImpl(df)
