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
            self.build_df(dataset),
            index=index,
            label=label,
            exclude=exclude,
        )

    def build_df(self, dataset: Dataset | list[Dataset]) -> DataFrame:
        """
        データセットから特徴量を生成する抽象クラス。
        単数か複数のdatasetから、featureのdataframeを組み上げる。
        """
        raise NotImplementedError("feature/build_df is not implemented.")

    def merge_df(self) -> DataFrame:
        if not isinstance(self.dataset, list):
            # 単数のdatasetの場合
            return concat(
                [self.dataset.df, self.df],
                axis=1,
                join="inner",
            )
        else:
            # 複数のdatasetの場合
            # TODO: 複数のdatasetを結合する処理を実装する
            raise NotImplementedError("まだ未実装")
