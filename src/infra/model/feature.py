from pandas import DataFrame, concat
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType

from infra.model.dataset import Dataset


class Feature(Dataset):
    """
    特徴量を表すクラス

    特徴量の元となるデータセットとしてBASEを持つ。
    BASEがあることで、バックテストなどの機能を十全に使える。
    """

    SCHEMA: dict[str, PdType]
    BASE: Dataset

    def __init__(self, df: DataFrame) -> None:
        super().__init__(df)
        self.base = self.__class__.BASE

    @property
    def merge_df(self) -> DataFrame:
        return concat(
            [self.base.df, self.df],
            axis=1,
            join="inner",
        )
