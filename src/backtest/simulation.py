#!/usr/bin/env python3
"""
LGBMモデルとLag特徴量を使用したバックテストシミュレーションを実行するスクリプト
"""
from backtesting import Backtest
from lightgbm import Booster
from pandas import DataFrame

# ドメイン知識のインポート (パスを調整)
from domain.dataset.common import read_df
from domain.dataset.ohlcv import Ohlcv
from domain.feature.lag import LagCloses10
from domain.model.lgbm import train_model_lgbm_lag_closes10 # モデル訓練関数
from domain.strategy.lag_lgbm import LagLgbmStrategy # 戦略クラス

# --- ヘルパー関数 ---
def create_ohlcv_from_data(name: str) -> Ohlcv:
    """
    データファイルからOhlcvオブジェクトを作成する
    """
    df = read_df(name)
    return Ohlcv(df)

def train_lag_model(ohlcv: Ohlcv) -> Booster:
    """
    OhlcvデータからLagCloses10特徴量を作成し、LGBMモデルを訓練する
    """
    lag_feature = LagCloses10(ohlcv)
    # domain.model.lgbm から訓練関数を呼び出す
    return train_model_lgbm_lag_closes10(lag_feature)

def run_backtest(df: DataFrame, model: Booster) -> None:
    """
    与えられたデータフレームと訓練済みモデルでバックテストを実行する
    """
    # バックテストの設定
    backtest = Backtest(
        df,
        LagLgbmStrategy, # domain.strategy から戦略クラスをインポート
        cash=100000,
        commission=0.002,
        exclusive_orders=True,
        trade_on_close=True,
        hedging=False,
    )

    # バックテスト実行（modelパラメータを渡す）
    # LagLgbmStrategyの__init__ (または基底クラスの__init__) に渡される
    stats = backtest.run(model=model)

    # 結果の表示
    print("--- バックテスト結果 ---")
    print(stats)
    print("----------------------")

    # プロットの表示
    try:
        backtest.plot()
        print("バックテストのプロットを表示しました。")
    except Exception as e:
        print(f"プロットの表示中にエラーが発生しました: {e}")
        print("プロットを表示するには、適切なグラフィカル環境が必要です。")

# --- メイン実行部分 ---
def main():
    """
    データ読み込み、モデル訓練、バックテスト実行のメインフロー
    """
    print("--- シミュレーション開始 ---")
    dataset_name = "aapl" # データセット名を指定
    print(f"{dataset_name.upper()}データを読み込み中...")
    try:
        ohlcv = create_ohlcv_from_data(dataset_name)
    except FileNotFoundError:
        print(f"エラー: データファイルが見つかりません。src/domain/dataset/data/{dataset_name.upper()}.csv を確認してください。")
        return
    except Exception as e:
        print(f"データ読み込み中にエラーが発生しました: {e}")
        return

    print("LGBMモデルを訓練中...")
    try:
        model = train_lag_model(ohlcv)
        print("モデル訓練完了。")
    except Exception as e:
        print(f"モデル訓練中にエラーが発生しました: {e}")
        return

    print("バックテスト実行中...")
    try:
        # バックテストには元のOHLCVデータフレームを渡す
        run_backtest(ohlcv.df, model)
    except Exception as e:
        print(f"バックテスト実行中にエラーが発生しました: {e}")
        return

    print("--- シミュレーション完了 ---")

if __name__ == "__main__":
    # このファイルが直接実行された場合にmain関数を呼び出す
    # (例: python src/backtest/simulation.py)
    main()
