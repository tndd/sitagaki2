import numpy as np

# 外部ライブラリbacktestingからStrategyをインポート
from backtesting import Strategy as BacktestStrategy
from lightgbm import Booster


class Strategy(BacktestStrategy):
    """
    backtestingライブラリを使用した戦略の基底クラス
    """

    # 必要に応じて共通機能を追加
    model: Booster = None

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def get_feature(self) -> np.ndarray:
        """
        特徴量を取得するメソッド。
        派生クラスでオーバーライドする必要がある。
        """
        raise NotImplementedError("Subclasses must implement this method.")
