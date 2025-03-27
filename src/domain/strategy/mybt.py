import math

import numpy as np
import yfinance as yf

from domain.feature.common import SCALE_BP
from domain.model.lgbm import train_model_lgbm_closes_n4
from fixture.factory.feature.closes import factory_closes_n4_cycle


# NumPy配列から安全に要素を取得する関数
def safe_get(arr, index):
    """NumPy配列から安全に値を取得する"""
    # スカラー値として要素を取り出す
    item = arr.item(index) if hasattr(arr, "item") else arr[index]
    return float(item)


# 特徴量生成関数
def get_feature(closes, position):
    """
    4日分の過去の値動きから特徴量を生成する

    Args:
        closes: 終値の配列
        position: 現在の位置

    Returns:
        numpy.ndarray: 特徴量の配列
    """
    if position < 4:
        return None

    # 安全にNumPy配列から値を取得
    now_val = safe_get(closes, position)
    lag1_val = safe_get(closes, position - 1)
    lag2_val = safe_get(closes, position - 2)
    lag3_val = safe_get(closes, position - 3)
    lag4_val = safe_get(closes, position - 4)

    # モデルが期待する値動きの形式（対数収益率）に変換
    now = math.log(now_val / lag1_val) * SCALE_BP
    lag_1 = math.log(lag1_val / lag2_val) * SCALE_BP
    lag_2 = math.log(lag2_val / lag3_val) * SCALE_BP
    lag_3 = math.log(lag3_val / lag4_val) * SCALE_BP

    # 2次元配列として返す
    return np.array([[now, lag_1, lag_2, lag_3]])


# シグナル生成関数
def generate_signal(closes, model, position):
    """
    モデルを使用して売買シグナルを生成する

    Args:
        closes: 終値の配列
        model: 学習済みLightGBMモデル
        position: 現在の位置

    Returns:
        int: 1（買い）、-1（売り）、0（何もしない）
    """
    f = get_feature(closes, position)
    if f is None:
        return 0

    # モデルが返す予測値を使用
    pred = model.predict(f)[0]
    if pred > 0:
        return 1
    elif pred < 0:
        return -1
    else:
        return 0


# シンプルなバックテスト実行関数（プロット機能なし）
def run_simple_backtest_no_plot(
    data, model, initial_cash=100000.0, commission=0.0, size_pct=0.5
):
    """
    シンプルなバックテストを実行する関数（backtraderを使わず、プロットも行わない）

    Args:
        data: バックテスト用データ (pandas DataFrame)
        model: 学習済みLightGBMモデル
        initial_cash: 初期資金
        commission: 手数料率
        size_pct: ポジションサイズの割合

    Returns:
        dict: バックテストの結果
    """
    # データの準備
    if isinstance(data, str):
        # ティッカーシンボルが渡された場合、データをダウンロード
        df = yf.download(data, start="2020-01-01")
    else:
        # データフレームが渡された場合、そのまま使用
        df = data.copy()

    # 終値の配列を取得
    closes = df["Close"].values

    # バックテストの初期設定
    cash = initial_cash
    position = 0  # 0: なし、1: 買い、-1: 売り
    position_size = 0
    trades = []

    # バックテストのメインループ
    for i in range(5, len(closes)):  # 5から開始（4日分のデータが必要なため）
        # シグナルの生成
        signal = generate_signal(closes, model, i)

        # 現在の価格（安全に取得）
        current_price = safe_get(closes, i)

        # 現在のポジションを解消
        if position != 0:
            # ポジションの価値を計算
            position_value = position_size * current_price
            # 手数料を計算
            commission_value = position_value * commission
            # キャッシュを更新
            cash = cash + position_value - commission_value
            # トレード記録を追加
            if len(trades) > 0:
                trades.append(
                    {
                        "exit_date": df.index[i],
                        "exit_price": current_price,
                        "profit_loss": position_value
                        - trades[-1]["position_value"]
                        - commission_value,
                    }
                )
            position = 0
            position_size = 0

        # 新しいポジションを取る
        if signal != 0:
            # 使用可能な資金の計算
            available_cash = cash * size_pct
            # 購入可能な株数の計算
            shares = available_cash / current_price
            # 手数料を計算
            commission_value = available_cash * commission
            # キャッシュを更新
            cash = cash - available_cash - commission_value
            # ポジションを更新
            position = signal
            position_size = shares if signal > 0 else shares
            # トレード記録を追加
            trades.append(
                {
                    "entry_date": df.index[i],
                    "entry_price": current_price,
                    "position": "LONG" if signal > 0 else "SHORT",
                    "shares": shares,
                    "position_value": shares * current_price,
                    "commission": commission_value,
                }
            )

    # 最終ポジションを解消
    if position != 0:
        current_price = safe_get(closes, -1)  # 安全に取得
        position_value = position_size * current_price
        commission_value = position_value * commission
        cash = cash + position_value - commission_value
        if len(trades) > 0:
            trades.append(
                {
                    "exit_date": df.index[-1],
                    "exit_price": current_price,
                    "profit_loss": position_value
                    - trades[-1]["position_value"]
                    - commission_value,
                }
            )

    # 結果の生成
    stats = {
        "initial_value": initial_cash,
        "final_value": cash,
        "profit_loss": cash - initial_cash,
        "return_pct": (cash / initial_cash - 1) * 100,
        "num_trades": len(trades) // 2 if len(trades) > 0 else 0,
    }

    return stats


if __name__ == "__main__":
    # Google株価データ（代替として、Yahoo Financeからデータを取得）
    ticker = "GOOG"

    # バックテストの実行
    closes = factory_closes_n4_cycle()
    model = train_model_lgbm_closes_n4(closes)

    print("モデルのトレーニングが完了しました")

    # Googleのデータをダウンロード
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

    print(f"{ticker}のデータをダウンロードしました（{len(data)}行）")

    # バックテストの実行（backtraderを使わない単純な実装）
    stats = run_simple_backtest_no_plot(
        data, model, initial_cash=100000.0, commission=0.001, size_pct=0.5
    )

    # 結果の表示
    print("\nバックテスト結果:")
    print(f"初期資金: ${stats['initial_value']:.2f}")
    print(f"最終資金: ${stats['final_value']:.2f}")
    print(f"損益: ${stats['profit_loss']:.2f}")
    print(f"リターン: {stats['return_pct']:.2f}%")
    print(f"トレード回数: {stats['num_trades']}")
