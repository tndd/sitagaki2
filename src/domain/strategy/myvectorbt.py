import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import Booster

from domain.feature.closes.service import calc_feature_closes
from domain.model.lgbm import train_model_lgbm_closes_n4
from fixture.factory.feature.closes import factory_closes_n4_cycle


class VectorbtLikeBacktester:
    """
    vectorbtライクなバックテスト機能を提供するクラス
    """

    def __init__(self, data, init_cash=100000, fees=0.0):
        """
        Args:
            data: バックテスト用のデータ（pd.DataFrame形式、'Close'カラムが必要）
            init_cash: 初期資金
            fees: 取引手数料
        """
        self.data = data.copy()
        self.init_cash = init_cash
        self.fees = fees

        # 結果用の変数を初期化
        self.cash = init_cash
        self.positions = 0
        self.values = []
        self.trades = []
        self.dates = []

    def run_from_signals(self, entries, exits, size=1.0, short_entries=False):
        """
        売買シグナルに基づいてバックテストを実行

        Args:
            entries: エントリーシグナル（True/Falseのシリーズ）
            exits: エグジットシグナル（True/Falseのシリーズ）
            size: ポジションサイズ（0-1の値）
            short_entries: ショートエントリーを許可するかどうか

        Returns:
            self: バックテスト結果を含むインスタンス
        """
        prices = self.data["Close"]

        # 初期化
        self.cash = self.init_cash
        self.positions = 0
        self.values = [self.init_cash]  # 初期値を設定
        self.trades = []
        self.dates = [self.data.index[0]]  # 最初の日付

        # 各日付でバックテストを実行
        for i in range(1, len(prices)):
            price = prices.iloc[i]
            date = prices.index[i]

            # 現在のポジション価値
            position_value = self.positions * price

            # エグジットが発生した場合（ポジションを保有している場合のみ）
            if exits.iloc[i] and self.positions != 0:
                entry_price = 0
                entry_date = date
                if self.trades and "entry_price" in self.trades[-1]:
                    entry_price = self.trades[-1]["entry_price"]
                    entry_date = self.trades[-1]["entry_date"]

                # ショートポジションの場合
                if self.positions < 0:
                    # ショート決済による利益計算 (買い戻し)
                    pnl = abs(self.positions) * (entry_price - price)
                    self.cash += abs(position_value) + pnl  # 証拠金と利益を戻す
                else:
                    # ロングポジションの決済
                    self.cash += position_value

                # トレード記録
                self.trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": (price - entry_price) * self.positions,
                        "type": "long" if self.positions > 0 else "short",
                    }
                )

                # ポジションのリセット
                self.positions = 0

            # エントリーが発生した場合（ポジションを保有していない場合のみ）
            if entries.iloc[i] and self.positions == 0:
                # 購入する数量（現金の半分を使用）
                position_size = (self.cash * size) / price

                # ショート・ロングの処理
                if short_entries:
                    # ショートポジション
                    self.positions = -position_size  # 負の値でショートを表現
                else:
                    # ロングポジション
                    self.cash -= position_size * price  # 購入による現金減少
                    self.positions = position_size  # 正の値でロングを表現

                # トレード記録
                self.trades.append(
                    {
                        "entry_date": date,
                        "entry_price": price,
                        "type": "short" if short_entries else "long",
                        "size": abs(self.positions),
                    }
                )

            # ポートフォリオ価値の計算
            if self.positions >= 0:
                # ロングポジションまたはポジションなし
                portfolio_value = self.cash + position_value
            else:
                # ショートポジション
                portfolio_value = self.cash  # 証拠金は既にcashに含まれる

            self.values.append(portfolio_value)
            self.dates.append(date)

        return self

    def value(self):
        """
        ポートフォリオ価値のシリーズを返す
        """
        return pd.Series(self.values, index=self.dates)

    def returns(self):
        """
        リターンのシリーズを返す
        """
        values = self.value()
        return values.pct_change().fillna(0)

    def stats(self):
        """
        パフォーマンス統計を計算
        """
        values = self.value()
        returns = self.returns()

        # 基本的なパフォーマンス指標を計算
        total_return = (values.iloc[-1] / self.init_cash - 1) * 100

        # 年率リターン（252営業日で計算）（異常値の場合は0を返す）
        try:
            days = len(values)
            annual_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100
            if np.isnan(annual_return) or np.isinf(annual_return):
                annual_return = 0
        except:
            annual_return = 0

        # シャープレシオ（無リスク金利は0と仮定）
        try:
            sharpe_ratio = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() != 0
                else 0
            )
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                sharpe_ratio = 0
        except:
            sharpe_ratio = 0

        # ドローダウン計算
        peak = values.cummax()
        drawdown = (values / peak - 1) * 100
        max_drawdown = drawdown.min()

        # 勝率計算
        completed_trades = [trade for trade in self.trades if "pnl" in trade]
        win_trades = sum(1 for trade in completed_trades if trade["pnl"] > 0)
        total_completed_trades = len(completed_trades)
        win_rate = (
            (win_trades / total_completed_trades * 100)
            if total_completed_trades > 0
            else 0
        )

        stats = {
            "初期資金": self.init_cash,
            "最終価値": values.iloc[-1],
            "トータルリターン(%)": round(total_return, 2),
            "年率リターン(%)": round(annual_return, 2),
            "シャープレシオ": round(sharpe_ratio, 2),
            "最大ドローダウン(%)": round(max_drawdown, 2),
            "取引回数": len(self.trades),
            "完了した取引": total_completed_trades,
            "勝率(%)": round(win_rate, 2),
        }

        return stats


def run_vectorbt_like_backtest(data, model=None, size=0.5):
    """
    vectorbtライクなバックテスターを使用した戦略の実装

    Args:
        data: バックテスト用のデータ（pd.DataFrame形式、'Close'カラムが必要）
        model: 予測モデル（LightGBM Boosterモデル）
        size: ポジションサイズ（デフォルト0.5 = 50%）

    Returns:
        dict: バックテスト結果
    """
    if model is None:
        # モデルが指定されていない場合は訓練する
        closes = factory_closes_n4_cycle()
        model = train_model_lgbm_closes_n4(closes)

    # 特徴量の作成
    features = create_features(data)

    # 予測値の計算
    predictions = model.predict(features)

    # 売買シグナルの生成
    buy_signals = pd.Series(False, index=data.index)
    sell_signals = pd.Series(False, index=data.index)

    # 最初の4日分はデータ不足のため取引なし
    for i in range(len(predictions)):
        if i + 4 < len(data):
            if predictions[i] > 0:  # 上昇予測
                buy_signals.iloc[i + 4] = True
            elif predictions[i] < 0:  # 下降予測
                sell_signals.iloc[i + 4] = True

    # バックテスト実行
    backtester_long = VectorbtLikeBacktester(data, init_cash=100000, fees=0.0)
    backtester_long.run_from_signals(entries=buy_signals, exits=sell_signals, size=size)

    backtester_short = VectorbtLikeBacktester(data, init_cash=100000, fees=0.0)
    backtester_short.run_from_signals(
        entries=sell_signals, exits=buy_signals, size=size, short_entries=True
    )

    # 結果表示
    print("===== ロングストラテジーのパフォーマンス =====")
    long_stats = backtester_long.stats()
    for key, value in long_stats.items():
        print(f"{key}: {value}")

    print("\n===== ショートストラテジーのパフォーマンス =====")
    short_stats = backtester_short.stats()
    for key, value in short_stats.items():
        print(f"{key}: {value}")

    # 組み合わせパフォーマンス
    long_values = backtester_long.value()
    short_values = backtester_short.value()
    combined_values = long_values + short_values - 100000  # 初期資金を1回分引く
    combined_returns = combined_values.pct_change().fillna(0)

    # 統計指標を安全に計算
    try:
        total_return = (combined_values.iloc[-1] / (2 * 100000 - 100000) - 1) * 100
        if np.isnan(total_return) or np.isinf(total_return):
            total_return = 0
    except:
        total_return = 0

    try:
        annual_return = (
            (1 + total_return / 100) ** (252 / len(combined_values)) - 1
        ) * 100
        if np.isnan(annual_return) or np.isinf(annual_return):
            annual_return = 0
    except:
        annual_return = 0

    try:
        sharpe_ratio = (
            combined_returns.mean() / combined_returns.std() * np.sqrt(252)
            if combined_returns.std() != 0
            else 0
        )
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            sharpe_ratio = 0
    except:
        sharpe_ratio = 0

    print("\n===== 組み合わせ戦略のパフォーマンス =====")
    print(f"トータルリターン(%): {total_return:.2f}")
    print(f"年率リターン(%): {annual_return:.2f}")
    print(f"シャープレシオ: {sharpe_ratio:.2f}")
    print(f"取引総数: {len(backtester_long.trades) + len(backtester_short.trades)}")

    return {
        "backtester_long": backtester_long,
        "backtester_short": backtester_short,
        "long_values": long_values,
        "short_values": short_values,
        "combined_values": combined_values,
        "long_stats": long_stats,
        "short_stats": short_stats,
    }


def create_features(data):
    """
    4日分の過去の値動きから特徴量を作成

    Args:
        data: バックテスト用のデータ（pd.DataFrame形式、'Close'カラムが必要）

    Returns:
        features: 特徴量の配列
    """
    if len(data) < 4:
        return np.array([])

    closes = data["Close"].values
    features = []

    for i in range(3, len(closes)):
        # 現在の価格と過去3日分の価格を取得
        now = closes[i]
        lag_1 = closes[i - 1]
        lag_2 = closes[i - 2]
        lag_3 = closes[i - 3]

        features.append([now, lag_1, lag_2, lag_3])

    return np.array(features)


def plot_performance(result):
    """
    バックテスト結果をプロット

    Args:
        result: バックテスト結果
    """
    plt.figure(figsize=(12, 10))

    # ポートフォリオ価値のプロット
    plt.subplot(2, 1, 1)
    plt.plot(result["long_values"], label="Long Strategy")
    plt.plot(result["short_values"], label="Short Strategy")
    plt.plot(result["combined_values"], label="Combined Strategy")
    plt.title("Portfolio Value")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # ドローダウンのプロット
    plt.subplot(2, 1, 2)

    # 各戦略のドローダウン計算
    long_dd = (result["long_values"] / result["long_values"].cummax() - 1) * 100
    short_dd = (result["short_values"] / result["short_values"].cummax() - 1) * 100
    combined_dd = (
        result["combined_values"] / result["combined_values"].cummax() - 1
    ) * 100

    plt.fill_between(
        long_dd.index, long_dd.values, 0, color="blue", alpha=0.3, label="Long Drawdown"
    )
    plt.fill_between(
        short_dd.index,
        short_dd.values,
        0,
        color="red",
        alpha=0.3,
        label="Short Drawdown",
    )
    plt.fill_between(
        combined_dd.index,
        combined_dd.values,
        0,
        color="green",
        alpha=0.3,
        label="Combined Drawdown",
    )
    plt.title("Drawdown (%)")
    plt.ylabel("Drawdown %")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return plt.gcf()


if __name__ == "__main__":
    # Backtestingライブラリに組み込まれているGOOGのテストデータを使用
    from backtesting.test import GOOG

    # pandas DataFrameに変換
    data = pd.DataFrame(GOOG)

    # モデルの準備
    closes = factory_closes_n4_cycle()
    model = train_model_lgbm_closes_n4(closes)

    # バックテストの実行
    result = run_vectorbt_like_backtest(data, model=model)

    # 結果のプロット
    plot_performance(result)
