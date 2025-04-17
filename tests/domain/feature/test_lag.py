import pytest
from pandas import DataFrame

from domain.feature.lag import derive_lag_df_from_ohlcv
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
    assert lag.field.index == "Date"
    assert lag.field.label == ["l0"]
    assert lag.field.exclude == []
    assert lag.field.col_names == [
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
    # 10件+label分の列数
    assert lag.df.shape[1] == 11
    # Nから10件+label分を引いた行数
    assert lag.df.shape[0] == N - 11
    # ohlcvと結合前と結合後で行数が変わらないことを確認
    assert lag.df_with_ohlcv.shape[0] == lag.df.shape[0]


@pytest.mark.parametrize("n", [1, 10, 100])
def test_derive_lag_df_from_ohlcv_normal(n):
    """
    ohlcvからlag特徴量を抽出する機能のテスト。
    ここでは正常系を検証する。(n>0)
    """
    ohlcv = factory_ohlcv_random_walk()
    lag_df = derive_lag_df_from_ohlcv(ohlcv, n)
    assert isinstance(lag_df, DataFrame)
    # label分の1を加えた0~nまでの個数
    assert lag_df.shape[1] == n + 1
    # dropnaのせいで厳密一致することはないが、最新のindexは一致する
    assert lag_df.index[-100:].equals(ohlcv.df.index[-100:])
    # 特徴量dfの長さは0でない
    assert lag_df.shape[0] > 0


@pytest.mark.parametrize("n", [0, -1, -10])
def test_derive_lag_df_from_ohlcv_abnormal(n):
    """
    ohlcvからlag特徴量を抽出する機能のテスト。
    ここでは異常系を検証する。(n<=0)
    """
    n_cols = 1000
    ohlcv = factory_ohlcv_random_walk(n_cols)
    lag_df = derive_lag_df_from_ohlcv(ohlcv, n)
    assert isinstance(lag_df, DataFrame)
    assert lag_df.shape[0] == n_cols
    # lagが作られる前に値が返されるので、列数は0となる
    assert lag_df.shape[1] == 0
    # dropnaのせいで厳密一致することはないが、最新のindexは一致する
    assert lag_df.index[-100:].equals(ohlcv.df.index[-100:])
