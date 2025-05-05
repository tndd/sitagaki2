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
    データの中身は1.0で埋められただけのもの。
    """
    df = DataFrame(
        1.0,
        columns=list(OHLCV_FEATURE_IMPL_DEFINITION.keys()),
    )
    return OhlcvFeatureImpl(df)
