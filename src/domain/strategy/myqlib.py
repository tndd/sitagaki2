import os
from datetime import datetime

# 日本語表示の警告を非表示にする設定
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# qlib関連の警告も非表示にする
import logging

logging.getLogger("qlib").setLevel(logging.ERROR)

# モジュール読み込み時の警告メッセージを一時的にリダイレクト
import sys
from io import StringIO


# 警告メッセージをキャプチャする関数
def silence_warnings():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    return old_stdout


# 元の出力に戻す関数
def restore_stdout(old_stdout):
    sys.stdout = old_stdout


# QLIBの初期化
def init_qlib():
    """
    QLIBを初期化し、サンプルデータが存在しない場合はダウンロードします
    """
    # データパスの設定
    provider_uri = "~/.qlib/qlib_data/cn_data"
    data_path = os.path.expanduser(provider_uri)

    # データが存在しない場合はダウンロード
    if not os.path.exists(data_path):
        print("サンプルデータが見つかりません。ダウンロードします...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        from qlib.tests.data import GetData

        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    # QLIBの初期化（警告を非表示にして実行）
    old_stdout = silence_warnings()
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    restore_stdout(old_stdout)
    print("QLIBの初期化が完了しました")


# 基本的なデータアクセスのサンプル
def basic_data_access():
    """
    基本的なデータアクセスのサンプルを示します
    """
    print("\n基本的なデータアクセスのデモ:")

    try:
        # データのアクセス方法を表示
        print("利用可能な指標(フィールド)の例:")
        print("$close: 終値")
        print("$open: 始値")
        print("$high: 高値")
        print("$low: 安値")
        print("$volume: 出来高")
        print("$vwap: 出来高加重平均価格")

        print("\nフィールド演算の例:")
        print("$close/$open - 1: 日内リターン")
        print("Ref($close, -1)/$close - 1: 前日比リターン")
        print("Mean($close, 5): 5日間の終値平均")
        print("$high - $low: 値幅")

        print("\n注意: 実際のデータ取得にはサンプルデータが必要です")
        print("必要に応じてサンプルデータのダウンロードを行ってください")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


# モメンタム投資戦略クラスの実装
class MomentumStrategy:
    def __init__(self, lookback_days=20, topk=5):
        """
        モメンタム戦略の初期化

        Args:
            lookback_days (int): 過去のリターン計算期間
            topk (int): 選択する上位銘柄数
        """
        self.lookback_days = lookback_days
        self.topk = topk

    def generate_signals(self, instruments, start_date, end_date):
        """
        モメンタム因子に基づいて投資シグナルを生成

        Args:
            instruments (list): 銘柄リスト
            start_date (str): 開始日
            end_date (str): 終了日

        Returns:
            dict: 日付ごとの選択銘柄リスト
        """
        print(
            f"モメンタム戦略: 過去{self.lookback_days}日間のリターンで上位{self.topk}銘柄を選択"
        )

        # モメンタム因子の定義
        momentum_factor = f"Ref($close, -{self.lookback_days})/$close - 1"

        try:
            # 因子データの取得
            factor_data = D.features(
                instruments, [momentum_factor], start_date, end_date
            )
            print(f"取得した因子データの形状: {factor_data.shape}")

            # 各日付ごとの選択銘柄
            selected_stocks = {}

            # 日付リストの取得
            dates = sorted(factor_data.index.get_level_values("datetime").unique())

            # 各日付で因子値に基づいて銘柄を選択
            for date in dates:
                try:
                    # その日の因子データを取得
                    date_data = factor_data.loc[pd.IndexSlice[:, date], :]

                    # 欠損値を除外
                    valid_data = date_data.dropna()

                    if len(valid_data) > 0:
                        # 因子値でソート（降順）
                        ranked_stocks = valid_data.sort_values(
                            by=momentum_factor, ascending=False
                        )

                        # 上位銘柄を選択
                        top_stocks = ranked_stocks.index.get_level_values(
                            "instrument"
                        ).tolist()[: min(self.topk, len(ranked_stocks))]
                        selected_stocks[date] = top_stocks
                    else:
                        selected_stocks[date] = []
                except Exception as e:
                    print(f"日付 {date} の処理でエラー: {e}")
                    selected_stocks[date] = []

            return selected_stocks

        except Exception as e:
            print(f"シグナル生成エラー: {e}")
            return {}

    def backtest(self, instruments, start_date, end_date):
        """
        戦略のバックテスト（簡易版）

        Args:
            instruments (list): 銘柄リスト
            start_date (str): バックテスト開始日
            end_date (str): バックテスト終了日

        Returns:
            dict: バックテスト結果
        """
        # シグナル生成
        signals = self.generate_signals(instruments, start_date, end_date)

        if not signals:
            print("シグナルを生成できませんでした")
            return {}

        # バックテスト結果の表示
        print(f"\nバックテスト期間: {start_date} ～ {end_date}")
        print(f"銘柄数: {len(instruments)}")
        print(f"シグナル数: {len(signals)}")

        # 最初の数日のシグナルを表示
        sample_dates = list(signals.keys())[:5]
        print("\n選択銘柄サンプル (最初の5営業日):")
        for date in sample_dates:
            stocks = signals[date]
            print(f"{date.date()}: {stocks}")

        return signals


# サンプル銘柄の取得
def get_sample_stocks():
    try:
        # CSI300指数構成銘柄を取得
        instruments = list(D.instruments(market="csi300"))

        if len(instruments) > 0:
            # 最大30銘柄を使用
            sample_stocks = instruments[:30]
            print(f"サンプル銘柄数: {len(sample_stocks)}")
            return sample_stocks
        else:
            # データが取得できない場合はダミー銘柄を使用
            print("銘柄データが取得できません。ダミー銘柄を使用します。")
            return ["SH000001", "SH600000", "SZ399001", "SZ000001"]

    except Exception as e:
        print(f"銘柄取得エラー: {e}")
        # ダミー銘柄を返す
        return ["SH000001", "SH600000", "SZ399001", "SZ000001"]


# 銘柄のパフォーマンス可視化
def plot_stock_performance(stock_id, start_date, end_date):
    try:
        # 株価データを取得
        fields = ["$close"]
        stock_data = D.features([stock_id], fields, start_date, end_date)

        if len(stock_data) > 0:
            # チャートを作成
            plt.figure(figsize=(10, 6))
            plt.plot(
                stock_data.index.get_level_values("datetime"),
                stock_data["$close"].values,
            )
            plt.title(f"{stock_id} 株価推移")
            plt.ylabel("株価")
            plt.xlabel("日付")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{stock_id}_price.png")
            print(f"{stock_id}の株価チャートを保存しました: {stock_id}_price.png")
            return True
        else:
            print(f"{stock_id}のデータが取得できませんでした")
            return False
    except Exception as e:
        print(f"チャート作成エラー: {e}")
        return False


# モメンタム投資戦略のシミュレーション
def simulate_momentum_strategy():
    print("\n--- モメンタム投資戦略のシミュレーション ---")

    # サンプルポートフォリオの構築
    # 注: 実際の運用ではさらに複雑な計算が必要です

    # 戦略の説明
    print("【モメンタム戦略の概要】")
    print("1. 過去のリターンに基づいて銘柄をランク付け")
    print("2. モメンタムの強い上位銘柄を選択して投資")
    print("3. 定期的にポートフォリオをリバランス")

    # 戦略パラメータの例
    lookback_days = 20  # 過去20日間のリターン
    topk = 5  # 上位5銘柄を選択

    # サンプルポートフォリオのパフォーマンス (シミュレーション用)
    dates = pd.date_range(start="2018-01-01", end="2018-06-30", freq="B")  # 営業日

    # シミュレーション結果サンプル作成
    np.random.seed(42)  # 再現性のため

    # 累積リターン計算用の初期値
    initial_value = 100
    daily_returns = np.random.normal(
        0.0007, 0.015, len(dates)
    )  # 平均0.07%、標準偏差1.5%の日次リターン

    # 累積リターンの計算
    portfolio_values = [initial_value]
    for ret in daily_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    portfolio_values = portfolio_values[1:]  # 初期値を除外

    # ベンチマーク (市場平均) のシミュレーション
    benchmark_returns = np.random.normal(
        0.0004, 0.01, len(dates)
    )  # 平均0.04%、標準偏差1.0%
    benchmark_values = [initial_value]
    for ret in benchmark_returns:
        benchmark_values.append(benchmark_values[-1] * (1 + ret))
    benchmark_values = benchmark_values[1:]

    # 結果の可視化
    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio_values, label="Momentum Strategy")
    plt.plot(dates, benchmark_values, label="Benchmark", linestyle="--")
    plt.title("Momentum Strategy vs Benchmark (Simulation)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("momentum_strategy_simulation.png")
    print(f"シミュレーションチャートを保存しました: momentum_strategy_simulation.png")

    # パフォーマンス指標の計算
    final_return = (portfolio_values[-1] / initial_value - 1) * 100
    benchmark_return = (benchmark_values[-1] / initial_value - 1) * 100

    print("\n【シミュレーション結果】")
    print(f"期間: 2018-01-01 ～ 2018-06-30")
    print(f"モメンタム戦略リターン: {final_return:.2f}%")
    print(f"ベンチマークリターン: {benchmark_return:.2f}%")
    print(f"超過リターン: {final_return - benchmark_return:.2f}%")

    # リスク指標
    strategy_volatility = np.std(daily_returns) * np.sqrt(252) * 100  # 年率換算
    benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252) * 100
    sharpe_ratio = (np.mean(daily_returns) * 252) / (
        np.std(daily_returns) * np.sqrt(252)
    )

    print("\n【リスク指標】")
    print(f"戦略ボラティリティ(年率): {strategy_volatility:.2f}%")
    print(f"ベンチマークボラティリティ(年率): {benchmark_volatility:.2f}%")
    print(f"シャープレシオ: {sharpe_ratio:.2f}")

    return {
        "dates": dates,
        "portfolio_values": portfolio_values,
        "benchmark_values": benchmark_values,
        "final_return": final_return,
        "benchmark_return": benchmark_return,
    }


# QLIBで使える表現の例
def qlib_expressions_examples():
    print("\n--- QLIBで使える表現の例 ---")

    expressions = [
        ("$close", "終値"),
        ("$open", "始値"),
        ("$high", "高値"),
        ("$low", "安値"),
        ("$volume", "出来高"),
        ("$factor", "株式分割調整係数"),
        ("$vwap", "出来高加重平均価格"),
        ("Ref($close, -1)", "1日前の終値"),
        ("Ref($close, -5)", "5日前の終値"),
        ("Mean($close, 5)", "5日間の終値の平均"),
        ("Std($close, 5)", "5日間の終値の標準偏差"),
        ("Max($high, 10)", "過去10日間の高値の最大値"),
        ("Min($low, 10)", "過去10日間の安値の最小値"),
        ("$close/$open - 1", "日中リターン"),
        ("Ref($close, -1)/$close - 1", "前日比リターン"),
        ("Ref($close, -5)/$close - 1", "5日前比リターン"),
        ("Mean($close, 5)/Mean($close, 20) - 1", "短期/長期移動平均乖離率"),
        ("Corr($close, $volume, 10)", "価格と出来高の10日相関"),
        ("Rsquare($close, $open, 10)", "始値と終値の10日決定係数"),
        ("Resi($close, Mean($close, 5), 10)", "5日移動平均からの乖離"),
    ]

    print("QLIBでは以下のような式を使って株価データを分析できます：")
    print("\n{:<35} | {:<}".format("表現", "意味"))
    print("-" * 60)
    for expr, desc in expressions:
        print("{:<35} | {:<}".format(expr, desc))

    print("\nこれらの表現を組み合わせて、様々な投資戦略を実装できます。")


# メイン関数
def main():
    print("=== PyQLIBを使った投資戦略シミュレーションのデモ ===\n")

    try:
        # PyTorchの初期警告をキャプチャ
        old_stdout = silence_warnings()
        # ダミー操作でPyTorchの警告を出させる
        import qlib.contrib.model

        restore_stdout(old_stdout)

        # QLIBの初期化
        print("【ステップ1】QLIBの初期化")
        init_qlib()

        # QLIBで使える表現の例
        print("\n【ステップ2】QLIBで使える表現")
        qlib_expressions_examples()

        # モメンタム戦略のシミュレーション
        print("\n【ステップ3】モメンタム戦略のシミュレーション")
        result = simulate_momentum_strategy()

        print("\nデモが完了しました！")
        print(
            "注意: このサンプルは学習目的のみで、実際の投資判断には使用しないでください。"
        )
        print(
            "\n詳細なドキュメントは https://qlib.readthedocs.io/en/latest/ を参照してください。"
        )

    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
