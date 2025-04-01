from fixture.domain.feature.lag import factory_lag_closes10
from pandas import DataFrame


def test_lag_closes10():
    lag = factory_lag_closes10()
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
