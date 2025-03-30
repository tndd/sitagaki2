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

    def __init__(
        self,
        dataset: Dataset | list[Dataset],
        index: str | None = None,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        self.dataset = dataset
        super().__init__(
            self.derive_df(dataset),
            index=index,
            label=label,
            exclude=exclude,
        )

    @property
    def merge_df(self) -> DataFrame:
        return concat(
            [self.dataset.df, self.df],
            axis=1,
            join="inner",
        )

    @staticmethod
    def derive_df(dataset: Dataset | list[Dataset]) -> DataFrame:
        """
        データセットから特徴量を生成する抽象クラス。
        単数か複数のdatasetから、featureのdataframeを組み上げる。
        """
        raise NotImplementedError("feature/derive_df is not implemented.")
