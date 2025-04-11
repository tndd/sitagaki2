import numpy as np

# 外部ライブラリbacktestingからStrategyをインポート
from backtesting import Strategy as BacktestStrategy
from lightgbm import Booster


class Strategy(BacktestStrategy):
    """
    backtestingライブラリを使用した戦略の基底クラス
    """

    # 必要に応じて共通機能を追加
    model: Booster = None  # モデルを保持するクラス変数

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        # initでmodelを受け取る (backtesting > 0.3.3)
        # backtestingライブラリはparams経由で渡されたものを自動でインスタンス変数に割り当てる
        # そのため、明示的に self.model = params['model'] とする必要はないことが多い
        # ただし、型ヒントや明示性のために書いておくことも可能
        if "model" in params:
            self.model = params["model"]

    def get_feature(self) -> np.ndarray:
        """
        特徴量を取得するメソッド。
        派生クラスでオーバーライドする必要がある。
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # backtesting < 0.3.3 用の互換性コードは削除 (最新版を想定)
