from pandas import DataFrame, concat
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType

from infra.model.dataset import Dataset


class Feature(Dataset):
    """
    特徴量を表すクラス

    特徴量の元となるデータセットとしてdatasetを持つ。
    datasetを受け取り、特徴量DFを生成するところまで行う。
    """

    SCHEMA: dict[str, PdType]

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        super().__init__(self.derive_df(dataset.df))

    @property
    def merge_df(self) -> DataFrame:
        return concat(
            [self.dataset.df, self.df],
            axis=1,
            join="inner",
        )

    @staticmethod
    def derive_df(df: DataFrame) -> DataFrame:
        """
        データセットから特徴量を生成する抽象クラス。
        initにて渡されたdfから特徴量を生成する。
        """
        raise NotImplementedError("feature/derive_df is not implemented.")
