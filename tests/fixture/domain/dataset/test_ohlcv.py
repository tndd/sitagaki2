import numpy as np
from pandas import DataFrame

from domain.dataset.ohlcv import Ohlcv
from fixture.domain.dataset.ohlcv import factory_ohlcv, factory_ohlcv_random_walk


def test_factory_ohlcv():
    ohlcv = factory_ohlcv()
    assert isinstance(ohlcv, Ohlcv)


def test_factory_ohlcv_random_walk():
    """
    random walkで生成されたデータの基本的な性質を検証するテスト

    以下の点を確認:
        1. データの品質（欠損値なし、適切な型）
        2. データの範囲（現実的な値の範囲内）
        3. 統計的な性質（平均、標準偏差）
        4. 時系列としての性質（隣接する値間の関係）

    特に、対数差分を使用した時系列データとして期待される性質を満たしているかを確認
    """
    n = 1000
    ohlcv = factory_ohlcv_random_walk(n=n)
    assert isinstance(ohlcv, Ohlcv)
    df = ohlcv.df

    # データフレームの基本チェック
    assert isinstance(df, DataFrame)
    assert not df.isna().any().any()  # 欠損値がないことを確認
    assert len(df) == n  # 指定した長さのデータが生成されていることを確認

    # データ型のチェック
    for col in ["Open", "High", "Low", "Close"]:
        assert np.issubdtype(df[col].dtype, np.floating)  # 浮動小数点型であることを確認
    assert np.issubdtype(df["Volume"].dtype, np.integer)  # 整数型であることを確認

    # データの範囲チェック
    # 対数差分は通常±1000BP以内に収まる
    close = df["Close"]
    log_returns = np.log(close / close.shift(1)) * 10000
    assert log_returns.min() > -1000
    assert log_returns.max() < 1000

    # 基本的な統計量のチェック
    # 対数差分の平均は0に近いはず
    assert abs(log_returns.mean()) < 10  # 10BP以内に収まることを確認
    # 標準偏差は現実的な範囲内
    assert 0 < log_returns.std() < 100  # 100BP以内に収まることを確認

    # 時系列の連続性チェック
    # 隣接する対数差分間で負の相関があることを確認
    log_returns_next = log_returns.shift(1)
    correlation = log_returns.corr(log_returns_next)
    assert correlation < 0  # 負の相関があることを確認
    assert correlation > -1  # 完全な負の相関ではないことを確認
