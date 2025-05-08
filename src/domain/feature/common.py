from functools import cached_property

from pandas import DataFrame
from pandera.api.pandas.types import (
    PandasDtypeInputTypes as PanderaType,
)

from domain.dataset.ohlcv import OHLCV_DEFINITION, OHLCV_INDEX
from infra.model.dataset import LabeledDataset


class OhlcvFeature(LabeledDataset):
    """
    ohlcvを元として生成される特徴量を表すクラス。

    Props:
        df:
            ohlcvと特徴量のカラムを持つデータフレーム。
            definitionに従ったカラムのDFが返される。
        definition:
            dfのカラム型定義。
            クラスとしての基本的なdefinitionにはohlcvを含める。

            *definition_feature:
                dfの特徴量カラム型定義。
                ohlcvを除きたい場合もあるために用意されている。
        label:
            dfのラベルカラム名。

    ohlcvを要素として持つ理由:
        バックテストで特徴量のパフォーマンスの検証を行うため。
        ohlcvが無いと、具体的な価格の推移を計算できないから。
    """

    def __init__(
        self,
        df: DataFrame,
        definition: dict[str, PanderaType],
        label: str,
    ) -> None:
        self.definition_feature = definition
        super().__init__(
            df=df,
            definition=OHLCV_DEFINITION | definition,
            label=label,
            index=OHLCV_INDEX,
        )

    @property
    def columns_feature_and_label(self) -> list[str]:
        """
        特徴量とラベルのカラム名を返す。
        """
        return list(self.definition_feature.keys())

    @cached_property
    def columns_feature(self) -> list[str]:
        """
        特徴量のみのカラム名を返す。
        毎回ループ処理が走らないよう、念の為cached_propertyを使う。
        """
        return [
            col
            for col in self.definition_feature.keys()
            if col != self.label
        ]

    @property
    def df_feature(self) -> DataFrame:
        """
        特徴量のみのデータフレームを返す。
        学習済みのモデルに与えるための値として使う。
        """
        return self.df.loc[:, self.columns_feature]
