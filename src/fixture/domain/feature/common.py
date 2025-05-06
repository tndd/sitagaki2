from pandas import DataFrame, date_range

from domain.dataset.ohlcv import OHLCV_DEFINITION
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
    """
    NUM_ROWS = 100
    columns = list(OHLCV_FEATURE_IMPL_DEFINITION.keys())
    data = {
        name: [float(f"{index}.{i + 1}") for i in range(NUM_ROWS)]
        for index, name in enumerate(columns)
    }
    # ohlcvカラムとDateカラムを追加
    ohlcv_data = {
        "Open": [1.0] * NUM_ROWS,
        "High": [1.0] * NUM_ROWS,
        "Low": [1.0] * NUM_ROWS,
        "Close": [1.0] * NUM_ROWS,
        "Volume": [1] * NUM_ROWS,
        "Date": date_range(start="2023-01-01", periods=NUM_ROWS),
    }
    df = DataFrame(data | ohlcv_data)
    return OhlcvFeatureImpl(df)


if __name__ == "__main__":
    # 臨時: 動作確認用
    ohlcv_feature = factory_ohlcv_feature_impl()
    print(ohlcv_feature.df)
