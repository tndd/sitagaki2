from pandas import DataFrame
from pandera import Column, DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType


class Dataset:
    """
    dataframeを扱うための抽象クラス
    ラベル付きデータセットとそうでないデータセット両方を想定してる。

    Properties:
        df: DataFrame
            dataframeはここに格納される。

        field: Field
            フィールド定義を管理する。

    ClassProperties:
        SCHEMA: dict[str, PdType]
            フィールド定義を管理する。
            indexはスキーマ情報には含めない。
    """

    SCHEMA: dict[str, PdType]

    def __init__(
        self,
        df: DataFrame,
        index: str | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """
        注意: labelとexcludeの入力型について
            入力の段階では、str, list, Noneの3つを取り得るが、
            内部的(Fieldクラス)は一貫してlistとして扱う。
        """
        # スキーマの定義と検証
        self.field = Field(
            definition=self.__class__.SCHEMA,
            index=index,
            exclude=exclude,
        )
        self.field.schema.validate(df)
        # 検証されたDataFrameを受け入れ
        self.df: DataFrame = df
        # 指定のインデックスが指定されてなければ、indexを設定する
        if isinstance(index, str) and self.df.index.name != index:
            self.df = self.df.set_index(index)


class Field:
    """
    Dataframeのフィールド定義を管理するクラス
    dict型の定義を受け取り、カラム名やschemaなど柔軟な形式で返す。

    注意:
        indexについてはdefinitionから除外される。

    Properties:
        index: str | None
            インデックス名を指定する。
            インデックスがない場合は、Noneを指定する。

        exclude: list[str]
            特徴量としては含めない項目を指定する。
            想定としては、Dateのような日付データなど。
    """

    def __init__(
        self,
        definition: dict[str, PdType],
        index: str | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        # indexが指定されている場合、definitionから除外
        self.definition: dict[str, PdType] = {
            k: v for k, v in definition.items() if k != index
        }
        # Index: Noneが設定される場合もある
        self.index = index
        # Exclude: list[str]の形式に変換される
        if exclude is None:
            self.exclude = []
        elif isinstance(exclude, str):
            self.exclude = [exclude]
        elif isinstance(exclude, list):
            self.exclude = exclude
        else:
            raise TypeError(f"不正なexclude => {exclude}")

    @property
    def defined_names(self) -> list[str]:
        """
        フィールド定義のカラム名を返す。
        definitionからindexのみ除外される。
        テーブルの全カラム名一覧を取得するイメージ。
        """
        return list(self.definition.keys())

    @property
    def names(self) -> list[str]:
        """
        インデックスと除外項目を除いたカラム名のリスト。
        基本的にはこちらを使う。
        """
        return self.label + self.exclude

    @property
    def feature_names(self) -> list[str]:
        """
        学習対象である特徴量に当たる部分のカラム名を返す。
        つまりインデックスとラベル、さらに除外指定されたカラムも除外される。
        """
        exclude_cols = set(self.label_exclude_names)
        return [col for col in self.definition if col not in exclude_cols]

    @property
    def schema(self) -> DataFrameSchema:
        """
        スキーマ検証用のためのDataFrameSchemaを返す。
        """
        schema_dict = {
            column_name: Column(dtype) for column_name, dtype in self.definition.items()
        }
        return DataFrameSchema(schema_dict)
