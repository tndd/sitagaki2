import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import Booster

from domain.feature.closes.service import calc_feature_closes
from domain.model.lgbm import train_model_lgbm_closes_n4
from fixture.factory.feature.closes import factory_closes_n4_cycle


class SimpleBacktester:
    """
    シンプルなバックテスト実装。自作のバックテストロジックを使用。
    """

    def __init__(self, data, initial_cash=100000, commission=0.0):
        """
        Args:
            data: バックテスト用のデータ（pd.DataFrame形式、'Close'カラムが必要）
            initial_cash: 初期資金
            commission: 取引手数料（%）
        """
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.positions = 0
        self.equity = []
        self.returns = []
        self.trades = []
        self.dates = []  # 日付を追跡

    def run(self, model, size=0.5):
        """
        バックテストを実行

        Args:
            model: 予測モデル
            size: ポジションサイズ（0.5 = 50%）

        Returns:
            pd.DataFrame: バックテスト結果
        """
        # 特徴量の作成
        features = self._create_features()

        if features is None or len(features) == 0:
            return None

        # 予測
        predictions = model.predict(features)

        # バックテストの実行
        self.cash = self.initial_cash
        self.positions = 0
        self.equity = []  # 空のリストから開始
        self.dates = []  # 日付を追跡

        # day_idxがデータの範囲内にあることを確認
        for i in range(len(predictions)):
            day_idx = i + 4  # 特徴量の開始位置（4日目から）

            # インデックスチェック
            if day_idx >= len(self.data):
                break

            # 日付を追跡
            self.dates.append(self.data.index[day_idx])
            close_price = self.data["Close"].iloc[day_idx]

            # 現在のポジションを清算
            if self.positions != 0:
                # ポジションを清算して現金化
                self.cash += self.positions * close_price * (1 - self.commission)
                self.positions = 0

            # 予測に基づいて新しいポジションを取る
            if predictions[i] > 0:  # 上昇予測
                # 買いポジション
                shares_to_buy = (self.cash * size) / close_price
                self.cash -= shares_to_buy * close_price * (1 + self.commission)
                self.positions += shares_to_buy
                self.trades.append(
                    {
                        "date": self.data.index[day_idx],
                        "type": "buy",
                        "price": close_price,
                        "shares": shares_to_buy,
                    }
                )
            elif predictions[i] < 0:  # 下降予測
                # 売りポジション（空売り）
                shares_to_sell = (self.cash * size) / close_price
                self.cash -= shares_to_sell * close_price * (1 + self.commission)
                self.positions -= shares_to_sell
                self.trades.append(
                    {
                        "date": self.data.index[day_idx],
                        "type": "sell",
                        "price": close_price,
                        "shares": shares_to_sell,
                    }
                )

            # ポートフォリオ価値の計算
            portfolio_value = self.cash + self.positions * close_price
            self.equity.append(portfolio_value)

        # 結果の整形（必要に応じてequityの長さを調整）
        if len(self.equity) > 0:
            # 追跡した日付を使用してインデックスを設定
            equity_series = pd.Series(self.equity, index=self.dates)
            self.returns = equity_series.pct_change().fillna(0)

            # 統計情報の計算
            stats = self._calculate_stats(equity_series)
            return stats
        else:
            # 十分なデータがない場合
            return {
                "初期資金": self.initial_cash,
                "最終資金": self.initial_cash,
                "トータルリターン(%)": 0,
                "年率リターン(%)": 0,
                "最大ドローダウン(%)": 0,
                "シャープレシオ": 0,
                "勝率(%)": 0,
                "取引回数": 0,
                "注意": "十分なデータがなくバックテストを完了できませんでした",
            }

    def _create_features(self):
        """
        4日分の過去の値動きから特徴量を作成

        Returns:
            features: 特徴量の配列
        """
        if len(self.data) < 4:
            return None

        closes = self.data["Close"].values
        features = []

        for i in range(3, len(closes)):
            # 現在の価格と過去3日分の価格を取得
            now = closes[i]
            lag_1 = closes[i - 1]
            lag_2 = closes[i - 2]
            lag_3 = closes[i - 3]

            features.append([now, lag_1, lag_2, lag_3])

        return np.array(features)

    def _calculate_stats(self, equity_series):
        """
        バックテスト結果の統計情報を計算

        Args:
            equity_series: エクイティカーブ

        Returns:
            dict: 統計情報
        """
        # リターンの計算
        returns = equity_series.pct_change().fillna(0)

        # 累積リターン
        total_return = (equity_series.iloc[-1] / self.initial_cash - 1) * 100

        # 年率リターン（252営業日で計算）
        days = len(equity_series)
        annual_return = (
            ((1 + total_return / 100) ** (252 / days) - 1) * 100 if days > 0 else 0
        )

        # 最大ドローダウン
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # シャープレシオ（無リスク金利は0と仮定）
        sharpe_ratio = (
            np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        )

        # 勝率計算
        if len(self.trades) > 0:
            wins = sum(
                1
                for i in range(1, len(self.trades))
                if (
                    self.trades[i - 1]["type"] == "buy"
                    and self.trades[i]["price"] > self.trades[i - 1]["price"]
                )
                or (
                    self.trades[i - 1]["type"] == "sell"
                    and self.trades[i]["price"] < self.trades[i - 1]["price"]
                )
            )
            win_rate = (
                (wins / (len(self.trades) - 1)) * 100 if len(self.trades) > 1 else 0
            )
        else:
            win_rate = 0

        stats = {
            "初期資金": self.initial_cash,
            "最終資金": equity_series.iloc[-1],
            "トータルリターン(%)": round(total_return, 2),
            "年率リターン(%)": round(annual_return, 2),
            "最大ドローダウン(%)": round(max_drawdown, 2),
            "シャープレシオ": round(sharpe_ratio, 2),
            "勝率(%)": round(win_rate, 2),
            "取引回数": len(self.trades),
        }

        return stats

    def plot(self):
        """
        バックテスト結果をプロット
        """
        if len(self.equity) == 0:
            print("バックテストを先に実行してください")
            return

        # 追跡した日付を使用
        equity_series = pd.Series(self.equity, index=self.dates)

        plt.figure(figsize=(12, 8))

        # エクイティカーブ
        plt.subplot(2, 1, 1)
        plt.plot(equity_series)
        plt.title("Portfolio Value")
        plt.grid(True)

        # ドローダウン
        plt.subplot(2, 1, 2)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        plt.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
        plt.title("Drawdown (%)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return plt.gcf()


def run_closes_strategy_simple(data, model=None, size=0.5):
    """
    シンプルなバックテスト実装を使用したClosesStrategyの実行

    Args:
        data: バックテスト用のデータ（pd.DataFrame形式、'Close'カラムが必要）
        model: 予測モデル（LightGBM Boosterモデル）
        size: ポジションサイズ（デフォルト0.5 = 50%）

    Returns:
        backtester: バックテスターオブジェクト
        stats: バックテスト結果の統計情報
    """
    if model is None:
        # モデルが指定されていない場合は訓練する
        closes = factory_closes_n4_cycle()
        model = train_model_lgbm_closes_n4(closes)

    # バックテストの実行
    backtester = SimpleBacktester(data, initial_cash=100000, commission=0.0)
    stats = backtester.run(model, size=size)

    if stats is None:
        print("エラー: バックテストの実行中に問題が発生しました")
        return backtester, None

    return backtester, stats


if __name__ == "__main__":
    # Backtestingライブラリに組み込まれているGOOGのテストデータを使用
    from backtesting.test import GOOG

    # pandas DataFrameに変換
    data = pd.DataFrame(GOOG)

    # モデルの準備
    closes = factory_closes_n4_cycle()
    model = train_model_lgbm_closes_n4(closes)

    # バックテストの実行
    backtester, stats = run_closes_strategy_simple(data, model=model)

    # 結果表示
    if stats:
        print("バックテスト結果:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # グラフの表示
        backtester.plot()
