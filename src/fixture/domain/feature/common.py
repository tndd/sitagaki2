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
    ohlcvの値については、float=1.0, int=100, その他=Noneで埋める。
    """
    # 特徴量のダミーデータを作成
    feature_columns = list(OHLCV_FEATURE_IMPL_DEFINITION.keys())
    feature_data = {
        name: [
            index + i * 0.1 for i in range(100)
        ]  # 計算結果を直接floatとして生成
        for index, name in enumerate(feature_columns)
    }
    feature_df = DataFrame(feature_data)
    # OHLCVのダミーデータを作成
    ohlcv_data = {}
    for name, dtype in OHLCV_DEFINITION.items():
        if dtype is float:
            ohlcv_data[name] = [1.0] * 100
        elif dtype is int:
            ohlcv_data[name] = [100] * 100
        else:
            ohlcv_data[name] = [None] * 100  # その他の型はNoneで埋める
    ohlcv_df = DataFrame(ohlcv_data)
    # 特徴量DFとOHLCV DFを結合
    combined_df = ohlcv_df.join(feature_df)
    ## 'Date' カラムを追加し、インデックスに設定
    combined_df["Date"] = date_range(start="2023-01-01", periods=100)
    combined_df = combined_df.set_index("Date")
    return OhlcvFeatureImpl(combined_df)


if __name__ == "__main__":
    # 臨時: 動作確認用
    ohlcv_feature = factory_ohlcv_feature_impl()
    print(ohlcv_feature.df)
