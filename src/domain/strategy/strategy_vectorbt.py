import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from domain.dataset.ohlcv.service import read_df


def run_vectorbt_strategy(ticker="AAPL", stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    vectorbtを使った単純なバックテスト戦略の実装

    Args:
        ticker: 株式銘柄のティッカーシンボル (AAPL, AMD, SBUX のいずれか)
        stop_loss_pct: ストップロスの割合 (0.05 = 5%)
        take_profit_pct: 利確の割合 (0.10 = 10%)

    Returns:
        dict: バックテスト結果
    """
    # データの読み込み
    ohlcv = read_df(ticker)
    df = ohlcv.df.to_pandas()

    # 日付列をインデックスに設定
    df = df.set_index("Date")

    # 単純な移動平均線の計算
    fast_window = 20
    slow_window = 50
    fast_ma = vbt.MA.run(df["Close"], window=fast_window)
    slow_ma = vbt.MA.run(df["Close"], window=slow_window)

    # エントリーシグナルの生成：短期移動平均が長期移動平均を上回った時
    entries = fast_ma.ma_above(slow_ma)

    # イグジットシグナルの生成：短期移動平均が長期移動平均を下回った時
    exits = fast_ma.ma_below(slow_ma)

    # バックテストの実行
    pf = vbt.Portfolio.from_signals(
        df["Close"],
        entries,
        exits,
        sl_stop=stop_loss_pct,  # ストップロスを設定
        tp_stop=take_profit_pct,  # 利確を設定
        freq="1D",  # 日足データ
        init_cash=100000,  # 初期資金
        fees=0.001,  # 手数料 (0.1%)
        slippage=0.001,  # スリッページ (0.1%)
    )

    # パフォーマンス統計
    stats = pf.stats()

    # matplotlib用のプロット作成
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    # 価格と移動平均線
    df["Close"].plot(ax=axes[0], label="Close Price")
    fast_ma.ma.plot(ax=axes[0], label=f"MA({fast_window})")
    slow_ma.ma.plot(ax=axes[0], label=f"MA({slow_window})")

    # エントリーとイグジットポイントのプロット
    entries_idx = df.index[entries]
    exits_idx = df.index[exits]

    for idx in entries_idx:
        if idx in df.index:
            axes[0].axvline(x=idx, color="g", linestyle="--", alpha=0.5)

    for idx in exits_idx:
        if idx in df.index:
            axes[0].axvline(x=idx, color="r", linestyle="--", alpha=0.5)

    axes[0].set_title(f"{ticker} Price and Signals")
    axes[0].legend()

    # 資産推移
    pf.value().plot(ax=axes[1])
    axes[1].set_title("Portfolio Value")

    # ドローダウン
    # drawdownの値をパーセントに変換してプロット
    dd_series = pd.Series(-pf.drawdown().values.flatten() * 100, index=pf.value().index)
    dd_series.plot(ax=axes[2])
    axes[2].set_title("Drawdown (%)")
    axes[2].set_ylabel("Drawdown %")
    axes[2].grid(True)

    plt.tight_layout()

    # 結果レポート
    print(f"===== {ticker} の移動平均線クロス戦略のパフォーマンス =====")
    print(f"総リターン: {stats['Total Return [%]']:.2f}%")
    print(f"年率リターン: {stats.get('Annual Return [%]', 0):.2f}%")
    print(f"シャープレシオ: {stats['Sharpe Ratio']:.2f}")
    print(f"最大ドローダウン: {stats['Max Drawdown [%]']:.2f}%")
    print(f"勝率: {stats['Win Rate [%]']:.2f}%")
    print(f"取引回数: {stats['Total Trades']}")

    return {"stats": stats, "portfolio": pf, "figure": fig}


def run_rsi_strategy(
    ticker="AAPL",
    rsi_window=14,
    rsi_entry_low=30,
    rsi_entry_high=70,
    stop_loss_pct=0.05,
):
    """
    RSI指標を使ったバックテスト戦略

    Args:
        ticker: 株式銘柄のティッカーシンボル (AAPL, AMD, SBUX のいずれか)
        rsi_window: RSI計算用ウィンドウサイズ
        rsi_entry_low: RSI買いエントリーポイント（この値以下で買い）
        rsi_entry_high: RSI売りエントリーポイント（この値以上で売り）
        stop_loss_pct: ストップロスの割合

    Returns:
        dict: バックテスト結果
    """
    # データの読み込み
    ohlcv = read_df(ticker)
    df = ohlcv.df.to_pandas()

    # 日付列をインデックスに設定
    df = df.set_index("Date")

    # RSI指標の計算
    rsi = vbt.RSI.run(df["Close"], window=rsi_window)

    # エントリーシグナルの生成
    long_entries = rsi.rsi_below(rsi_entry_low)  # RSIが30以下で買い
    short_entries = rsi.rsi_above(rsi_entry_high)  # RSIが70以上で売り

    # イグジットシグナル（反対のシグナルが出たとき、またはRSIが中立域に戻ったとき）
    long_exits = rsi.rsi_above(50)  # RSIが50を上回ったら買いポジションを閉じる
    short_exits = rsi.rsi_below(50)  # RSIが50を下回ったら売りポジションを閉じる

    # ロングバックテスト
    long_pf = vbt.Portfolio.from_signals(
        df["Close"],
        long_entries,
        long_exits,
        sl_stop=stop_loss_pct,
        freq="1D",
        init_cash=100000,
        fees=0.001,
        slippage=0.001,
    )

    # ショートバックテスト
    short_pf = vbt.Portfolio.from_signals(
        df["Close"],
        short_entries,
        short_exits,
        short=True,  # ショート取引を有効化
        sl_stop=stop_loss_pct,
        freq="1D",
        init_cash=100000,
        fees=0.001,
        slippage=0.001,
    )

    # 結合ポートフォリオ（ロングとショートの組み合わせ）
    combined_pf = long_pf + short_pf

    # パフォーマンス統計
    long_stats = long_pf.stats()
    short_stats = short_pf.stats()
    combined_stats = combined_pf.stats()

    # matplotlib用のプロット作成
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # 価格チャート
    df["Close"].plot(ax=axes[0], label="Close Price")
    axes[0].set_title(f"{ticker} Price")

    # RSI指標
    rsi.rsi.plot(ax=axes[1], label="RSI")
    axes[1].axhline(
        y=rsi_entry_low,
        color="g",
        linestyle="--",
        label=f"RSI Buy Level ({rsi_entry_low})",
    )
    axes[1].axhline(
        y=rsi_entry_high,
        color="r",
        linestyle="--",
        label=f"RSI Sell Level ({rsi_entry_high})",
    )
    axes[1].axhline(y=50, color="k", linestyle="--", label="RSI Neutral (50)")
    axes[1].set_title("RSI Indicator")
    axes[1].set_ylim(0, 100)
    axes[1].legend()

    # 資産推移
    combined_pf.value().plot(ax=axes[2])
    axes[2].set_title("Combined Portfolio Value")

    # ロングとショートの資産推移
    long_pf.value().plot(ax=axes[2], label="Long Portfolio")
    short_pf.value().plot(ax=axes[2], label="Short Portfolio")
    axes[2].legend()

    # ドローダウン
    combined_dd = pd.Series(
        -combined_pf.drawdown().values.flatten() * 100, index=combined_pf.value().index
    )
    long_dd = pd.Series(
        -long_pf.drawdown().values.flatten() * 100, index=long_pf.value().index
    )
    short_dd = pd.Series(
        -short_pf.drawdown().values.flatten() * 100, index=short_pf.value().index
    )

    combined_dd.plot(ax=axes[3], label="Combined")
    long_dd.plot(ax=axes[3], label="Long")
    short_dd.plot(ax=axes[3], label="Short")
    axes[3].set_title("Drawdown (%)")
    axes[3].set_ylabel("Drawdown %")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()

    # 結果レポート
    print(f"===== {ticker} のRSI戦略のパフォーマンス =====")
    print(f"\n【ロング戦略】")
    print(f"総リターン: {long_stats['Total Return [%]']:.2f}%")
    print(f"年率リターン: {long_stats.get('Annual Return [%]', 0):.2f}%")
    print(f"シャープレシオ: {long_stats['Sharpe Ratio']:.2f}")
    print(f"最大ドローダウン: {long_stats['Max Drawdown [%]']:.2f}%")
    print(f"勝率: {long_stats['Win Rate [%]']:.2f}%")
    print(f"取引回数: {long_stats['Total Trades']}")

    print(f"\n【ショート戦略】")
    print(f"総リターン: {short_stats['Total Return [%]']:.2f}%")
    print(f"年率リターン: {short_stats.get('Annual Return [%]', 0):.2f}%")
    print(f"シャープレシオ: {short_stats['Sharpe Ratio']:.2f}")
    print(f"最大ドローダウン: {short_stats['Max Drawdown [%]']:.2f}%")
    print(f"勝率: {short_stats['Win Rate [%]']:.2f}%")
    print(f"取引回数: {short_stats['Total Trades']}")

    print(f"\n【組み合わせ戦略】")
    print(f"総リターン: {combined_stats['Total Return [%]']:.2f}%")
    print(f"年率リターン: {combined_stats.get('Annual Return [%]', 0):.2f}%")
    print(f"シャープレシオ: {combined_stats['Sharpe Ratio']:.2f}")
    print(f"最大ドローダウン: {combined_stats['Max Drawdown [%]']:.2f}%")
    print(f"取引回数: {combined_stats['Total Trades']}")

    return {
        "long_stats": long_stats,
        "short_stats": short_stats,
        "combined_stats": combined_stats,
        "portfolio": combined_pf,
        "figure": fig,
    }


def compare_strategies(ticker="AAPL"):
    """
    異なる戦略のパフォーマンスを比較する

    Args:
        ticker: 株式銘柄のティッカーシンボル

    Returns:
        dict: 比較結果
    """
    # データの読み込み
    ohlcv = read_df(ticker)
    df = ohlcv.df.to_pandas()

    # 日付列をインデックスに設定
    df = df.set_index("Date")

    # 各戦略の実行
    ma_result = run_vectorbt_strategy(ticker)
    rsi_result = run_rsi_strategy(ticker)

    # matplotlib用の比較プロット作成
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    # 資産推移を比較
    ma_result["portfolio"].value().plot(ax=axes[0], label="MA Strategy")
    rsi_result["portfolio"].value().plot(ax=axes[0], label="RSI Strategy")
    axes[0].set_title("Portfolio Values Comparison")
    axes[0].legend()
    axes[0].grid(True)

    # ドローダウン比較
    ma_dd = pd.Series(
        -ma_result["portfolio"].drawdown().values.flatten() * 100,
        index=ma_result["portfolio"].value().index,
    )
    rsi_dd = pd.Series(
        -rsi_result["portfolio"].drawdown().values.flatten() * 100,
        index=rsi_result["portfolio"].value().index,
    )

    ma_dd.plot(ax=axes[1], label="MA Strategy")
    rsi_dd.plot(ax=axes[1], label="RSI Strategy")
    axes[1].set_title("Drawdown Comparison (%)")
    axes[1].set_ylabel("Drawdown %")
    axes[1].legend()
    axes[1].grid(True)

    # リターン比較
    ma_returns = ma_result["portfolio"].returns()
    rsi_returns = rsi_result["portfolio"].returns()
    (ma_returns.cumsum() + 1).plot(ax=axes[2], label="MA Strategy")
    (rsi_returns.cumsum() + 1).plot(ax=axes[2], label="RSI Strategy")
    axes[2].set_title("Cumulative Returns")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    # 統計の比較
    print(f"===== {ticker} の戦略比較 =====")

    stats_comparison = pd.DataFrame(
        {
            "MA戦略": {
                "総リターン": ma_result["stats"]["Total Return [%]"],
                "年率リターン": ma_result["stats"].get("Annual Return [%]", 0),
                "シャープレシオ": ma_result["stats"]["Sharpe Ratio"],
                "最大ドローダウン": ma_result["stats"]["Max Drawdown [%]"],
                "勝率": ma_result["stats"]["Win Rate [%]"],
                "取引回数": ma_result["stats"]["Total Trades"],
            },
            "RSI戦略(組合せ)": {
                "総リターン": rsi_result["combined_stats"]["Total Return [%]"],
                "年率リターン": rsi_result["combined_stats"].get(
                    "Annual Return [%]", 0
                ),
                "シャープレシオ": rsi_result["combined_stats"]["Sharpe Ratio"],
                "最大ドローダウン": rsi_result["combined_stats"]["Max Drawdown [%]"],
                "勝率": rsi_result["combined_stats"]["Win Rate [%]"],
                "取引回数": rsi_result["combined_stats"]["Total Trades"],
            },
        }
    )

    print(stats_comparison)

    return {
        "ma_result": ma_result,
        "rsi_result": rsi_result,
        "comparison_figure": fig,
        "stats_comparison": stats_comparison,
    }


if __name__ == "__main__":
    # 単一戦略の実行例
    # ma_result = run_vectorbt_strategy("AAPL")
    # rsi_result = run_rsi_strategy("AAPL")

    # 戦略比較
    comparison = compare_strategies("AAPL")
    plt.show()
