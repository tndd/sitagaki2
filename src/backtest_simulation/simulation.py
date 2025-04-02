# 外部ライブラリbacktestingをインポート
from backtesting import Backtest
from lightgbm import Booster
from pandas import DataFrame

from domain.dataset.common import read_df
from domain.dataset.ohlcv import Ohlcv
from domain.feature.lag import LagCloses10
from domain.model.lgbm import train_model_lgbm_lag_closes10
from domain.strategy.lag_lgbm import LagLgbmStrategy


def create_ohlcv_from_data(name: str) -> Ohlcv:
    """
    データファイルからOhlcvオブジェクトを作成する
    """
    df = read_df(name)
    return Ohlcv(df)


def train_lag_model(ohlcv: Ohlcv) -> Booster:
    """
    LagCloses10特徴量を使ってLGBMモデルを訓練する
    """
    lag_feature = LagCloses10(ohlcv)
    return train_model_lgbm_lag_closes10(lag_feature)


def run_backtest(df: DataFrame, model: Booster) -> None:
    """
    バックテストの実行
    """
    # バックテストの設定
    backtest = Backtest(
        df,
        LagLgbmStrategy,
        cash=100000,  # 初期資金を増加（10,000 → 100,000）
        commission=0.002,  # 売買手数料
        exclusive_orders=True,  # 注文は同時に1つのみ
        trade_on_close=True,  # 終値で取引を実行
        hedging=False,  # ヘッジ無効
    )

    # バックテスト実行（modelパラメータを渡す）
    stats = backtest.run(model=model)  # sizeはStrategyクラスで定義済み

    # 結果の表示
    print(stats)

    # プロットの表示（グラフィカルな環境で実行する場合）
    backtest.plot()


def main():
    """
    メイン関数
    """
    print("AAPLデータを読み込み中...")
    ohlcv = create_ohlcv_from_data("aapl")

    print("LGBMモデルを訓練中...")
    model = train_lag_model(ohlcv)

    print("バックテスト実行中...")
    run_backtest(ohlcv.df, model)

    print("シミュレーション完了")


if __name__ == "__main__":
    main()
