import pytest
from pandas import DataFrame

from domain.dataset.ohlcv import OHLCV_INDEX
from domain.feature.lag import LAG_CLOSES_LABEL, derive_lag_df_from_ohlcv
from fixture.domain.dataset.ohlcv import factory_ohlcv_random_walk
from fixture.domain.feature.lag import factory_lag_closes10


def test_lag_closes10():
    """
    lag特徴量の基本的な性質を検証するテスト
    indexやlabel、カラム名が設定されているかという最低限の確認内容
    """
    N = 1000
    lag = factory_lag_closes10(N)
    # スキーマの定義チェック
    assert lag.index == OHLCV_INDEX
    assert lag.label == LAG_CLOSES_LABEL
    assert lag.columns_feature_and_label == [
        "l0",
        "l1",
        "l2",
        "l3",
        "l4",
        "l5",
        "l6",
        "l7",
        "l8",
        "l9",
        "l10",
    ]
    # columns_featureはlabelのl0が除かれている
    assert lag.columns_feature == [
        "l1",
        "l2",
        "l3",
        "l4",
        "l5",
        "l6",
        "l7",
        "l8",
        "l9",
        "l10",
    ]
    # dfのカラム数は10件+label分 + ohlcvのカラム数５件
    assert lag.df.shape[1] == 16
    # df_featureのカラム数は10件 (注意:labelは含まれていない)
    assert lag.df_feature.shape[1] == 10
    # 行数はNから10件+label分を引いた行数
    assert lag.df.shape[0] == N - 11
    # ohlcvカラムの有無に関わらず、行数は変わらないことを確認
    assert lag.df_feature.shape[0] == lag.df.shape[0]


@pytest.mark.parametrize("n_lags", [1, 10, 100])
def test_derive_lag_df_from_ohlcv_normal(n_lags):
    """
    ohlcvからlag特徴量を抽出する機能のテスト。
    ここでは正常系を検証する。(n_lags>0)
    """
    ohlcv = factory_ohlcv_random_walk()
    lag_df = derive_lag_df_from_ohlcv(ohlcv, n_lags)
    assert isinstance(lag_df, DataFrame)
    # label分の1を加えた0~nまでの個数
    assert lag_df.shape[1] == n_lags + 1
    # dropnaのせいで厳密一致することはないが、最新のindexは一致する
    assert lag_df.index[-100:].equals(ohlcv.df.index[-100:])
    # 特徴量dfの長さは0でない
    assert lag_df.shape[0] > 0


@pytest.mark.parametrize("n_lags", [0, -1, -10])
def test_derive_lag_df_from_ohlcv_abnormal(n_lags):
    """
    ohlcvからlag特徴量を抽出する機能のテスト。
    ここでは異常系を検証する。(n_lags<=0)

    異常系の入力には空のDataFrameが帰ってくることが期待される。
    """
    ohlcv = factory_ohlcv_random_walk()
    lag_df = derive_lag_df_from_ohlcv(ohlcv, n_lags)
    assert isinstance(lag_df, DataFrame)
    # 空のDataFrameであることを確認
    assert lag_df.shape == (0, 0)
