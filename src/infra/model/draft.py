from pandas import DataFrame
from pandera import Column, DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes as PanderaType


class Dataset:
    """
    dataframeを扱うための抽象クラス。
    dataframeごとの違いを明示的に扱うため。

    Props:
        df:             dataframeはここに格納される。
        definition:     カラム定義。単なるpythonの普遍的な型。
        index:          インデックスがあるならここで指定
    """

    def __init__(
        self,
        df: DataFrame,
        definition: dict[str, PanderaType],
        index: str | None = None,
    ) -> None:
        self.definition = definition
        self.index = index
        # スキーマ評価とdfの保持
        self.pandera_schema.validate(df)
        self.df = df
        # 指定のインデックスが指定されてなければ、indexを設定する
        if isinstance(index, str) and self.df.index.name != index:
            self.df = self.df.set_index(index)

    @property
    def pandera_schema(self) -> DataFrameSchema:
        """
        スキーマ検証用のためのDataFrameSchemaを返す。

        definitionは定義時点では単なるpythonの型。
        だがこの関数は、それをpannderaのカラム型に変換してDataFrameSchemaに加工して返す。
        """
        return DataFrameSchema(
            {
                name: Column(dtype)
                for name, dtype in self.definition.items()
            }
        )

    @property
    def columns(self) -> list[str]:
        return list(self.definition.keys())


class LabeledDataset(Dataset):
    """
    教師ありデータセットを表す抽象クラス
    教師ラベルは単数であるという前提

    Props:
        df: Dataframe
        definition: dict
        index: str
        label: str
    """

    def __init__(
        self,
        df: DataFrame,
        definition: dict[str, PanderaType],
        label: str,
        index: str | None = None,
    ) -> None:
        super().__init__(df, definition, index)
        self.label = label

    @property
    def non_label_columns(self) -> list[str]:
        """
        教師ラベル部分を除いたカラム名のリスト
        """
        return [col for col in self.columns if col != self.label]


class MultiLabeledDataset(Dataset):
    """
    WARN: 未使用クラス

    教師ありデータセットを表す抽象クラス
    教師ラベルは複数であるという前提

    Props:
        df: Dataframe
        definition: dict
        index: str
        labels: list[str]
    """

    def __init__(
        self,
        df: DataFrame,
        definition: dict[str, PanderaType],
        labels: list[str],
        index: str | None = None,
    ) -> None:
        super().__init__(df, definition, index)
        self.labels = labels

    @property
    def non_label_columns(self) -> list[str]:
        """
        教師ラベル部分を除いたカラム名のリスト
        """
        return [col for col in self.columns if col not in self.labels]
