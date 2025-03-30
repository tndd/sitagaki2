from dataclasses import dataclass

from pandas import DataFrame, concat

from domain.dataset.ohlcv2 import Ohlcv2
from infra.model.dataset import Dataset

SCALE_BP = 10000


@dataclass
class OhlcvFeature:
    """
    ohlcvを元として生成される特徴量を表すクラス

    ohlcvを要素として持つことでバックテスト時に、
    簡単に特徴量のパフォーマンスの検証が可能となる。
    """

    # TODO: テスト

    ohlcv: Ohlcv2
    feature: Dataset

    @property
    def merge_df(self) -> DataFrame:
        return concat(
            [self.ohlcv.df, self.feature.df],
            axis=1,
            join="inner",
        )
