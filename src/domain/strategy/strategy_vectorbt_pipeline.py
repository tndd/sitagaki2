import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from domain.dataset.ohlcv.service import read_df
from domain.strategy.strategy_vectorbt import run_rsi_strategy, run_vectorbt_strategy


def run_strategy_pipeline(
    tickers=["AAPL", "AMD", "SBUX"], strategy_type="ma", **params
):
    """
    複数の銘柄に対して選択した戦略を実行するパイプライン処理

    Args:
        tickers: 分析する銘柄のリスト
        strategy_type: 使用する戦略タイプ ("ma" または "rsi")
        **params: 戦略に渡すパラメータ

    Returns:
        dict: 各銘柄の結果を含む辞書
    """
    results = {}
    summary_stats = {}

    # 各銘柄に対して戦略を実行
    for ticker in tickers:
        print(f"\n実行中: {ticker} - {strategy_type}戦略")

        # 選択した戦略に基づいて実行
        if strategy_type == "ma":
            result = run_vectorbt_strategy(
                ticker=ticker,
                stop_loss_pct=params.get("stop_loss_pct", 0.05),
                take_profit_pct=params.get("take_profit_pct", 0.10),
            )
            stats = result["stats"]

        elif strategy_type == "rsi":
            result = run_rsi_strategy(
                ticker=ticker,
                rsi_window=params.get("rsi_window", 14),
                rsi_entry_low=params.get("rsi_entry_low", 30),
                rsi_entry_high=params.get("rsi_entry_high", 70),
                stop_loss_pct=params.get("stop_loss_pct", 0.05),
            )
            stats = result["combined_stats"]

        else:
            raise ValueError(f"不正な戦略タイプ: {strategy_type}")

        results[ticker] = result
        summary_stats[ticker] = {
            "総リターン": stats["total_return"],
            "年率リターン": stats["annual_return"],
            "シャープレシオ": stats["sharpe_ratio"],
            "最大ドローダウン": stats["max_drawdown"],
            "取引回数": stats["total_trades"],
        }

    # 結果の比較
    stats_df = pd.DataFrame(summary_stats).T
    print("\n===== 各銘柄のパフォーマンス比較 =====")
    print(stats_df)

    # パフォーマンス比較グラフ
    plot_performance_comparison(results, strategy_type)

    return {"results": results, "summary_stats": stats_df}


def plot_performance_comparison(results, strategy_type):
    """
    各銘柄のパフォーマンスを比較するプロット

    Args:
        results: 各銘柄の戦略結果
        strategy_type: 戦略タイプ
    """
    # 比較グラフの作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # 各銘柄のポートフォリオ価値を正規化して比較
    for ticker, result in results.items():
        if strategy_type == "ma":
            portfolio = result["portfolio"]
        else:  # RSI戦略の場合は組み合わせポートフォリオを使用
            portfolio = result["portfolio"]

        # 初期値で正規化
        normalized_value = portfolio.value() / portfolio.value().iloc[0]
        normalized_value.vbt.plot(ax=axes[0], label=f"{ticker}")

    axes[0].set_title("正規化したポートフォリオ価値の比較")
    axes[0].legend()
    axes[0].grid(True)

    # 各銘柄のドローダウン比較
    for ticker, result in results.items():
        if strategy_type == "ma":
            portfolio = result["portfolio"]
        else:
            portfolio = result["portfolio"]

        portfolio.drawdown().vbt.plot(ax=axes[1], label=f"{ticker}")

    axes[1].set_title("ドローダウンの比較")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def optimize_strategy_parameters(ticker="AAPL", strategy_type="ma", param_grid=None):
    """
    最適なパラメータを見つけるための戦略最適化

    Args:
        ticker: 銘柄のティッカーシンボル
        strategy_type: 最適化する戦略タイプ
        param_grid: 最適化するパラメータの範囲

    Returns:
        dict: 最適化結果
    """
    # データの読み込み
    ohlcv = read_df(ticker)
    df = ohlcv.df.to_pandas()
    df = df.set_index("Date")

    # デフォルトのパラメータグリッド
    if param_grid is None:
        if strategy_type == "ma":
            param_grid = {
                "fast_window": [10, 15, 20, 25],
                "slow_window": [40, 50, 60],
                "stop_loss_pct": [0.03, 0.05, 0.07],
                "take_profit_pct": [0.08, 0.10, 0.12],
            }
        elif strategy_type == "rsi":
            param_grid = {
                "rsi_window": [7, 14, 21],
                "rsi_entry_low": [25, 30, 35],
                "rsi_entry_high": [65, 70, 75],
                "stop_loss_pct": [0.03, 0.05, 0.07],
            }

    print(f"===== {ticker} の {strategy_type} 戦略最適化 =====")
    print(f"パラメータグリッド: {param_grid}")

    # 戦略タイプに基づいて最適化を実行
    if strategy_type == "ma":
        return optimize_ma_strategy(df, param_grid)
    elif strategy_type == "rsi":
        return optimize_rsi_strategy(df, param_grid)
    else:
        raise ValueError(f"不正な戦略タイプ: {strategy_type}")


def optimize_ma_strategy(df, param_grid):
    """
    移動平均戦略の最適化

    Args:
        df: データフレーム
        param_grid: パラメータグリッド

    Returns:
        dict: 最適化結果
    """
    # ベクトル化されたバックテストの準備
    fast_windows = param_grid["fast_window"]
    slow_windows = param_grid["slow_window"]
    stop_losses = param_grid["stop_loss_pct"]
    take_profits = param_grid["take_profit_pct"]

    # 移動平均を計算
    fast_mas = vbt.MA.run_combs(df["Close"], window=fast_windows)
    slow_mas = vbt.MA.run_combs(df["Close"], window=slow_windows)

    # すべての組み合わせのエントリーとイグジットシグナルを計算
    entries = fast_mas.ma_above(slow_mas)
    exits = fast_mas.ma_below(slow_mas)

    # パラメータの組み合わせを作成
    param_product = vbt.utils.params.create_param_product(
        {
            "fast_window": fast_windows,
            "slow_window": slow_windows,
            "sl_stop": stop_losses,
            "tp_stop": take_profits,
        }
    )

    # ポートフォリオをバックテスト
    pf = vbt.Portfolio.from_signals(
        df["Close"],
        entries,
        exits,
        sl_stop=param_product["sl_stop"],
        tp_stop=param_product["tp_stop"],
        freq="1D",
        init_cash=100000,
        fees=0.001,
        slippage=0.001,
    )

    # パフォーマンス指標を計算
    metrics = pf.stats()

    # 結果をDataFrameとして取得
    metrics_df = pd.DataFrame(
        {
            "fast_window": param_product["fast_window"],
            "slow_window": param_product["slow_window"],
            "stop_loss": param_product["sl_stop"],
            "take_profit": param_product["tp_stop"],
            "total_return": metrics["total_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["total_trades"],
        }
    )

    # シャープレシオでソート
    metrics_df = metrics_df.sort_values("sharpe_ratio", ascending=False)

    # 最良の結果を表示
    print("\n===== 最適化結果 (上位5件) =====")
    print(metrics_df.head(5))

    # 最良の組み合わせを使用して完全なバックテスト
    best_params = metrics_df.iloc[0].to_dict()

    return {
        "best_params": {
            "fast_window": int(best_params["fast_window"]),
            "slow_window": int(best_params["slow_window"]),
            "stop_loss_pct": float(best_params["stop_loss"]),
            "take_profit_pct": float(best_params["take_profit"]),
        },
        "metrics_df": metrics_df,
        "portfolio": pf,
    }


def optimize_rsi_strategy(df, param_grid):
    """
    RSI戦略の最適化

    Args:
        df: データフレーム
        param_grid: パラメータグリッド

    Returns:
        dict: 最適化結果
    """
    # ベクトル化されたバックテストの準備
    rsi_windows = param_grid["rsi_window"]
    rsi_lows = param_grid["rsi_entry_low"]
    rsi_highs = param_grid["rsi_entry_high"]
    stop_losses = param_grid["stop_loss_pct"]

    # 各RSIウィンドウでRSIを計算
    rsis = vbt.RSI.run_combs(df["Close"], window=rsi_windows)

    # 長期投資のためのシグナル（すべての組み合わせ）
    long_entries = rsis.rsi_below(rsi_lows)
    long_exits = rsis.rsi_above(50)

    # ショート投資のためのシグナル
    short_entries = rsis.rsi_above(rsi_highs)
    short_exits = rsis.rsi_below(50)

    # パラメータの組み合わせを作成
    param_product = vbt.utils.params.create_param_product(
        {
            "rsi_window": rsi_windows,
            "rsi_low": rsi_lows,
            "rsi_high": rsi_highs,
            "sl_stop": stop_losses,
        }
    )

    # ロングポートフォリオ
    long_pf = vbt.Portfolio.from_signals(
        df["Close"],
        long_entries,
        long_exits,
        sl_stop=param_product["sl_stop"],
        freq="1D",
        init_cash=100000,
        fees=0.001,
        slippage=0.001,
    )

    # ショートポートフォリオ
    short_pf = vbt.Portfolio.from_signals(
        df["Close"],
        short_entries,
        short_exits,
        short=True,
        sl_stop=param_product["sl_stop"],
        freq="1D",
        init_cash=100000,
        fees=0.001,
        slippage=0.001,
    )

    # 組み合わせポートフォリオ
    combined_pf = long_pf + short_pf

    # パフォーマンス指標を計算
    metrics = combined_pf.stats()

    # 結果をDataFrameとして取得
    metrics_df = pd.DataFrame(
        {
            "rsi_window": param_product["rsi_window"],
            "rsi_entry_low": param_product["rsi_low"],
            "rsi_entry_high": param_product["rsi_high"],
            "stop_loss": param_product["sl_stop"],
            "total_return": metrics["total_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "total_trades": metrics["total_trades"],
        }
    )

    # シャープレシオでソート
    metrics_df = metrics_df.sort_values("sharpe_ratio", ascending=False)

    # 最良の結果を表示
    print("\n===== 最適化結果 (上位5件) =====")
    print(metrics_df.head(5))

    # 最良の組み合わせを使用して完全なバックテスト
    best_params = metrics_df.iloc[0].to_dict()

    return {
        "best_params": {
            "rsi_window": int(best_params["rsi_window"]),
            "rsi_entry_low": int(best_params["rsi_entry_low"]),
            "rsi_entry_high": int(best_params["rsi_entry_high"]),
            "stop_loss_pct": float(best_params["stop_loss"]),
        },
        "metrics_df": metrics_df,
        "portfolio": combined_pf,
    }


if __name__ == "__main__":
    # 戦略パイプラインの実行例
    # pipeline_result = run_strategy_pipeline(strategy_type="ma")

    # 戦略最適化の実行例
    optimization_result = optimize_strategy_parameters(
        ticker="AAPL",
        strategy_type="rsi",
        param_grid={
            "rsi_window": [7, 14, 21],
            "rsi_entry_low": [25, 30, 35],
            "rsi_entry_high": [65, 70, 75],
            "stop_loss_pct": [0.03, 0.05, 0.07],
        },
    )

    best_params = optimization_result["best_params"]
    print(f"\n最適パラメータ: {best_params}")

    # 最適パラメータを使用した戦略実行
    optimized_result = run_rsi_strategy(ticker="AAPL", **best_params)

    plt.show()
