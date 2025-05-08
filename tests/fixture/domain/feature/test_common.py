from fixture.domain.feature.common import (
    OhlcvFeatureImpl,
    factory_ohlcv_feature_impl,
)


def test_factory_ohlcv_feature_impl():
    ohlcv_feature = factory_ohlcv_feature_impl()
    assert isinstance(ohlcv_feature, OhlcvFeatureImpl)
    # カラムはOF0~6だけでなく、頭にOHLCVのぶんも追加されてることを確認
    # 並びはohlcv -> featureの順
    assert ohlcv_feature.columns == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "OF0",
        "OF1",
        "OF2",
        "OF3",
        "OF4",
        "OF5",
        "OF6",
    ]
    # 特徴量 + ラベルのカラムについてのみであることを確認
    assert ohlcv_feature.feature_and_label_columns == [
        "OF0",
        "OF1",
        "OF2",
        "OF3",
        "OF4",
        "OF5",
        "OF6",
    ]
    # 特徴量カラムはOF1~6。OF0はラベルカラムのため含まれない
    assert ohlcv_feature.feature_columns == [
        "OF1",
        "OF2",
        "OF3",
        "OF4",
        "OF5",
        "OF6",
    ]
