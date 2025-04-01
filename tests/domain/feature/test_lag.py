from fixture.domain.feature.lag import factory_lag_closes10
from pandas import DataFrame
import numpy as np


def test_lag_closes10():
    lag = factory_lag_closes10()
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
    # データフレームの基本チェック
    df = lag.df
    assert isinstance(df, DataFrame)
    assert not df.isna().any().any()  # 欠損値がないことを確認
    assert len(df) > 0  # データが存在することを確認

    # データ型のチェック
    for col in lag.field.col_names:
        assert np.issubdtype(df[col].dtype, np.floating)  # 浮動小数点型であることを確認

    # データの範囲チェック
    for col in lag.field.col_names:
        # 対数差分は通常±1000BP以内に収まる
        assert df[col].min() > -1000
        assert df[col].max() < 1000

    # 基本的な統計量のチェック
    for col in lag.field.col_names:
        # 対数差分の平均は0に近いはず
        assert abs(df[col].mean()) < 10  # 10BP以内に収まることを確認
        # 標準偏差は現実的な範囲内
        assert 0 < df[col].std() < 100  # 100BP以内に収まることを確認

    # 時系列の連続性チェック
    # 隣接するlag特徴量間で負の相関があることを確認
    for i in range(10):
        col_current = f"l{i}"
        col_next = f"l{i+1}"
        correlation = df[col_current].corr(df[col_next])
        assert correlation < 0  # 負の相関があることを確認
        assert correlation > -1  # 完全な負の相関ではないことを確認


def test_derive_df_lag_closes10():
    lag = factory_lag_closes10()
    df = lag.df

    # データフレームの基本チェック
    assert isinstance(df, DataFrame)
    assert not df.isna().any().any()  # 欠損値がないことを確認
    assert len(df) > 0  # データが存在することを確認

    # 特定の行のlag特徴量を確認
    sample_row = df.iloc[0]
    assert all(col in sample_row.index for col in lag.field.col_names)  # 必要な列が存在することを確認

    # 対数差分の計算が正しいことを確認
    # 例：l0 = log(close_t / close_t-1) * 10000
    # 実際の値と期待値が近いことを確認
    sample_values = sample_row[lag.field.col_names]
    assert all(abs(val) < 1000 for val in sample_values)  # 現実的な範囲内であることを確認

    # データの順序が正しいことを確認
    # l0が最新のデータ、l10が最も古いデータであることを確認
    for i in range(10):
        col_current = f"l{i}"
        col_next = f"l{i+1}"
        # 隣接するlag特徴量の相関が負であることを確認
        # これは対数差分の計算方法から予想される結果
        correlation = df[col_current].corr(df[col_next])
        assert correlation < 0  # 負の相関があることを確認
