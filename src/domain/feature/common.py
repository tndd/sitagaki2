from typing import Protocol

from pandas import DataFrame, concat
from pandera.api.pandas.types import PandasDtypeInputTypes as PanderaType

from domain.dataset.ohlcv import Ohlcv
from infra.model.dataset import LabeledDataset


class OhlcvFeatureGenerator(Protocol):
    """
    ohlcvを元に特徴量を生成するための抽象クラス。

    ohlcv以外にも引数を受け取ることも想定しているため、
    クラス変数を取り得るので注意。
    """

    def generate(self, ohlcv: Ohlcv) -> DataFrame:
        """
        ohlcvから特徴量を生成する関数。
        """
        ...


class OhlcvFeature(LabeledDataset):
    """
    ohlcvを元として生成される特徴量を表すクラス。

    ohlcvを要素として持つ理由:
        バックテストで特徴量のパフォーマンスの検証を行うため。
        ohlcvが無いと、具体的な価格の推移を計算できないから。

    TODO: dfのカラム検討
        今のところdfのカラムはデフォルトでは特徴量の分しか持っていない。
        だが本当はohlcvのものもデフォルトで持たせて、
        必要に応じて特徴量のみに制限するという運用の方が適切ではないだろうか？
    """

    def __init__(
        self,
        ohlcv: Ohlcv,
        generator: OhlcvFeatureGenerator,
        definition: dict[str, PanderaType],
        label: str,
    ) -> None:
        """
        ohlcvを受け取り、特徴量を生成する。
        """
        self.ohlcv = ohlcv
        super().__init__(
            df=generator.generate(ohlcv),
            definition=definition,
            label=label,
            index="Date",
        )

    @property
    def df_with_ohlcv(self) -> DataFrame:
        """
        自身の特徴量dfとohlcv.dfを結合して返す。
        """
        return concat(
            [self.ohlcv.df, self.df],
            axis=1,
            join="inner",
        )

    @property
    def df_feature_with_ohlcv(self) -> DataFrame:
        """
        ohlcv含めてカラムを特徴量のみに絞って返す。
        バックテストで使われるのはこちらが想定される。
        """
        return concat(
            [
                self.ohlcv.df,
                self.df.loc[:, self.non_label_columns],
            ],
            axis=1,
            join="inner",
        )

    @property
    def df_feature(self) -> DataFrame:
        """
        特徴量のみのデータフレームを返す。
        学習済みのモデルに与えるための値として使う。
        """
        return self.df.loc[:, self.non_label_columns]
