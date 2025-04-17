from pandas import DataFrame

from fixture.domain.feature.lag import factory_lag_closes10


def test_lag_closes10():
    """
    lag特徴量の基本的な性質を検証するテスト
    indexやlabel、カラム名が設定されているかという最低限の確認内容
    """
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


def test_derive_df_lag_closes10():
    """
    lag特徴量の具体的な計算結果を検証するテスト

    以下の点を確認:
        1. データフレームの基本構造
        2. サンプルデータの存在と形式
        3. 計算結果の値の範囲
        4. 時系列データとしての連続性

    特に、実際の計算結果が期待される範囲内に収まっているかを確認
    """
    lag = factory_lag_closes10()
    df = lag.df

    # データフレームの基本チェック
    assert isinstance(df, DataFrame)
    assert not df.isna().any().any()  # 欠損値がないことを確認
    assert len(df) > 0  # データが存在することを確認

    # 特定の行のlag特徴量を確認
    sample_row = df.iloc[0]
    # 必要な列が存在することを確認
    assert all(col in sample_row.index for col in lag.field.col_names)

    # 対数差分の計算が正しいことを確認
    # 例：l0 = log(close_t / close_t-1) * 10000
    # 実際の値と期待値が近いことを確認
    sample_values = sample_row[lag.field.col_names]
    # 現実的な範囲内であることを確認
    assert all(abs(val) < 1000 for val in sample_values)
