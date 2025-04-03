import numpy as np
from backtesting import Backtest
from lightgbm import (
    Booster,
    early_stopping,
    train,
)
from pandas import DataFrame

# 必要なドメイン知識のインポート
from domain.dataset.common import read_df
from domain.dataset.ohlcv import Ohlcv
from domain.feature.common import OhlcvFeature
from domain.feature.lag import LagCloses10
# 外部ライブラリbacktestingからStrategyをインポート
from backtesting import Strategy as BacktestStrategy

# --- 基底戦略クラス (元 src/backtest/domain/strategy/base.py) ---
class Strategy(BacktestStrategy):
    """
    backtestingライブラリを使用した戦略の基底クラス
    """

    # 必要に応じて共通機能を追加
    model: Booster = None # モデルを保持するクラス変数（run_backtestで設定）

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        # initでmodelを受け取る (backtesting > 0.3.3)
        if hasattr(self, '_params') and 'model' in self._params:
             self.model = self._params['model']

    def get_feature(self) -> np.ndarray:
        """
        特徴量を取得するメソッド。
        派生クラスでオーバーライドする必要がある。
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # backtesting < 0.3.3 用の互換性コード (もし古いバージョンを使っている場合)
    # def init(self):
    #     if 'model' in self.params:
    #          self.model = self.params['model']


# --- LagLgbm戦略クラス (元 src/domain/strategy/lag_lgbm.py) ---
class LagLgbmStrategy(Strategy): # 継承元を上で定義したStrategyに変更
    """
    LagCloses10特徴量とLGBMモデルを使用したバックテスト戦略
    """

    # 予測しきい値 - この値を調整することで取引頻度を制御
    BUY_THRESHOLD = 10  # 買いポジションを取るしきい値を下げる
    SELL_THRESHOLD = -10  # 売りポジションを取るしきい値を下げる

    # バックテストのパラメータ
    size = 0.2  # ポジションサイズを増やす（資金の20%）

    def init(self):
        """
        戦略の初期化処理
        """
        # モデルはパラメータとして渡される (`run_backtest` で `model=model` として渡される)
        pass

    def get_feature(self) -> np.ndarray:
        """
        直近のデータからLagCloses10特徴量を作成
        """
        # バックテスト中の現在の時点までのデータ
        # DataFrameに変換（バックテストライブラリのデータ形式から変換）
        # self.data は backtesting ライブラリから提供されるデータオブジェクト
        df = DataFrame(self.data.df.iloc[: len(self.data)])
        # インデックス設定 (Ohlcvクラスが必要とするため)
        df.index.name = "Date"

        # データからOhlcv、そしてLagCloses10特徴量を作成
        ohlcv = Ohlcv(df)
        feature = LagCloses10(ohlcv)

        # 最新のデータポイントのみを特徴量として使用
        # ラベル列を除外
        latest_features = feature.df.iloc[-1:].drop(columns=[feature.field.label[0]])
        return latest_features.to_numpy()

    def next(self):
        """
        各バーごとに呼び出される取引ロジック
        """
        # 特徴量を準備
        try:
            # 十分なデータがあるかチェック (LagCloses10は最低12日分のデータが必要: shift(11)まで使うため)
            MIN_DATA_LEN = 12
            if len(self.data) < MIN_DATA_LEN:
                 # print(f"データ不足のためスキップ: 現在 {len(self.data)} < 必要 {MIN_DATA_LEN}")
                 return # データが足りない場合は何もしない

            features = self.get_feature()

            # モデルによる予測
            # self.model は init 時に backtesting フレームワーク経由で渡される
            prediction = self.model.predict(features)[0]

            # 現在のポジションをクローズ
            if self.position:
                self.position.close()

            # 予測値に基づいて取引
            if prediction > self.BUY_THRESHOLD:  # 上昇予測
                self.buy(size=self.size) # sizeパラメータを指定
            elif prediction < self.SELL_THRESHOLD:  # 下落予測
                self.sell(size=self.size) # sizeパラメータを指定

        except IndexError:
             # データが足りない場合などに発生する可能性がある
             print(f"データ不足のためスキップ: 現在のデータ長 {len(self.data)}")
        except Exception as e:
            # その他の予期せぬエラー
            print(f"エラー発生のためスキップ: {e}")


# --- モデル訓練関数 (元 src/domain/model/lgbm.py) ---
def train_model_lgbm_lag_closes10(dataset: OhlcvFeature) -> Booster:
    """
    LagCloses10特徴量データセットを使ってLightGBMモデルを訓練する
    """
    params = {
        "objective": "regression",  # 回帰問題として解く
        "metric": "rmse",  # 評価指標
        "boosting_type": "gbdt",  # 勾配ブースティング
        "num_leaves": 31,  # 木の複雑さ
        "learning_rate": 0.05,  # 学習率
        "feature_fraction": 0.9,  # 特徴選択の割合
        "bagging_fraction": 0.8,  # データのサブサンプリング割合
        "bagging_freq": 5,  # バギングの頻度
        "verbose": -1,  # ログを非表示
    }
    slt = dataset.get_split_labeled_tensor()
    lgb_train, lgb_test = slt.get_lgb_train_test()
    return train(
        params,
        lgb_train,
        valid_sets=[lgb_test],
        num_boost_round=500,  # ラウンド数
        callbacks=[early_stopping(50)],  # 早期停止の監視期間
    )


# --- ヘルパー関数 (元 src/backtest_simulation/simulation.py) ---
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
    return train_model_lgbm_lag_closes10(lag_feature)


def run_backtest(df: DataFrame, model: Booster) -> None:
    """
    与えられたデータフレームと訓練済みモデルでバックテストを実行する
    """
    # バックテストの設定
    backtest = Backtest(
        df,
        LagLgbmStrategy, # ここで定義した戦略クラスを使用
        cash=100000,  # 初期資金
        commission=0.002,  # 売買手数料
        exclusive_orders=True,  # 注文は同時に1つのみ
        trade_on_close=True,  # 終値で取引を実行
        hedging=False,  # ヘッジ無効
    )

    # バックテスト実行（modelパラメータを渡す）
    # LagLgbmStrategyのinitメソッドに渡され、self.modelとして利用可能になる
    stats = backtest.run(model=model)

    # 結果の表示
    print("--- バックテスト結果 ---")
    print(stats)
    print("----------------------")

    # プロットの表示（グラフィカルな環境で実行する場合）
    # 環境によっては表示されない、またはエラーになる可能性があります
    try:
        backtest.plot()
        print("バックテストのプロットを表示しました。")
    except Exception as e:
        print(f"プロットの表示中にエラーが発生しました: {e}")
        print("プロットを表示するには、適切なグラフィカル環境が必要です。")


# --- メイン実行部分 (元 src/backtest_simulation/simulation.py) ---
def main():
    """
    データ読み込み、モデル訓練、バックテスト実行のメインフロー
    """
    print("--- シミュレーション開始 ---")
    # データセット名を指定 (例: 'aapl', 'amd', 'sbux')
    dataset_name = "aapl"
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
        run_backtest(ohlcv.df, model)
    except Exception as e:
        print(f"バックテスト実行中にエラーが発生しました: {e}")
        return

    print("--- シミュレーション完了 ---")


if __name__ == "__main__":
    main()
