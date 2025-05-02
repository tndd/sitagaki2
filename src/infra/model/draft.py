from pandas import DataFrame
from pandera import Column, DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes as PanderaType


class Dataset:
    """
    dataframeを扱うための抽象クラス。
    dataframeごとの違いを明示的に扱うため。

    Props:
        df:             dataframeはここに格納される。
        definition:     カラム定義
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
        """
        schema_dict = {
            column_name: Column(dtype) for column_name, dtype in self.definition.items()
        }
        return DataFrameSchema(schema_dict)

    @property
    def columns(self) -> list[str]:
        return list(self.definition.keys())


class LabeledDataset(Dataset):
    """
    教師ありデータセットを表す抽象クラス

    Props:
        df: Dataframe
        definition: dict
        index: str
        label: list[str]
            どれが教師データであるかはここで指定する。
            教師データは基本的には単数を想定してるが、複数もあり得るためリスト形式
    """

    def __init__(
        self,
        df: DataFrame,
        definition: dict[str, PanderaType],
        index: str | None = None,
        label: str | list[str] | None = None,
    ) -> None:
        super().__init__(df, definition, index)
        # labelは必ず、list[str]の形式に変換される
        if label is None:
            self.label = []
        elif isinstance(label, str):
            self.label = [label]
        elif isinstance(label, list):
            self.label = label
        else:
            raise TypeError(f"不正なlabel => {label}")

    @property
    def non_label_columns(self) -> list[str]:
        """
        教師ラベル部分を除いたカラム名のリスト
        """
        return [col for col in self.columns if col not in self.label]
