from pandas import DataFrame

from domain.dataset.ohlcv import OHLCV_INDEX
from fixture.domain.feature.common import (
    OHLCV_FEATURE_IMPL_DEFINITION,
    factory_ohlcv_feature_impl,
)


def test_ohlcv_feature():
    """
    OhlcvFeature自体のテスト
    fixtureの方では行わなかった、インスタンスへの詳細なテストを行う
    """
    ohlcv_feature = factory_ohlcv_feature_impl()
    assert isinstance(ohlcv_feature.df, DataFrame)
    # インデックスはOHLCVのものがセットされることを確認
    assert ohlcv_feature.index == OHLCV_INDEX
    # 特徴量定義はdefinitionではなく"definition_feature"にセットされていることを確認
    assert ohlcv_feature.definition_feature == OHLCV_FEATURE_IMPL_DEFINITION
    # ラベルが設定されてる
    assert ohlcv_feature.label == "OF0"
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
    assert ohlcv_feature.columns_feature_and_label == [
        "OF0",
        "OF1",
        "OF2",
        "OF3",
        "OF4",
        "OF5",
        "OF6",
    ]
    # 特徴量カラムはOF1~6。OF0はラベルカラムのため含まれない
    assert ohlcv_feature.columns_feature == [
        "OF1",
        "OF2",
        "OF3",
        "OF4",
        "OF5",
        "OF6",
    ]
