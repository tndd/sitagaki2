from pandas import DataFrame
from pandera import Column, DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType
from sklearn.model_selection import train_test_split

from infra.model.tensor import LabeledTensor, SplitLabeledTensor


class Dataset:
    """
    dataframeを扱うための抽象クラス
    スキーマ定義と親の情報を持つ。

    SCHEMA:
        フィールド定義を管理する。
        indexはスキーマ情報には含めない。
    """

    SCHEMA: dict[str, PdType]

    def __init__(
        self,
        df: DataFrame,
        index: str | None = None,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """
        Args:
            df: DataFrame
                dataframeはここに格納される。

            index: str | None
                インデックス名を指定する。
                インデックスがない場合は、Noneを指定する。

            label: list[str] | str | None
                教師データのラベル名を指定する。
                ラベルがない場合は、何も入れない。

            exclude: list[str] | str | None
                特徴量としては含めない項目を指定する。
                想定としては、Dateのような日付データなど。

        Self:
            field: FieldDefinition
                フィールド定義を管理する。

        注意: labelとexcludeの入力型について
            入力の段階では、str, list, Noneの3つを取り得るが、
            内部的には一貫的にlistとして扱う。
        """
        # スキーマの定義と検証
        self.field = Field(
            definition=self.__class__.SCHEMA,
            index=index,
            label=label,
            exclude=exclude,
        )
        self.field.schema.validate(df)
        # 検証されたDataFrameを受け入れ
        self.df: DataFrame = df
        # Indexがあるなら設定
        if isinstance(index, str):
            self.df = self.df.set_index(index)

    def get_labeled_tensor(self) -> LabeledTensor:
        """
        教師ありデータのtensorに変換して返す
        """
        if len(self.field.label) == 0:
            # labels未定義状態で、この関数を呼んだ場合はエラーで落とす
            raise ValueError("There is no label in this schema.")
        elif len(self.field.label) == 1:
            # ラベルが１次元の場合
            return LabeledTensor(
                X=self.df.drop(
                    columns=self.field.label + self.field.exclude
                ).to_numpy(),
                y=self.df[self.field.label]
                .to_numpy()
                .ravel(),  # 1dラベルと確定しているので、ravelで警告を抑制
            )
        else:
            # 多次元ラベルの場合
            raise ValueError("WIP: Labelが複数の場合の動作は未定義")

    def get_split_labeled_tensor(
        self,
        test_size: int = 0.2,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> SplitLabeledTensor:
        """
        訓練用とテスト用にデータが分割された、
        教師ありデータのtensorを返す

        テストサイズは20%。
        時系列データが渡されることを考慮し、デフォルトではシャッフルはしない。
        """
        dssv = self.get_labeled_tensor()
        # 時系列を考慮したデータ分割
        X_train, X_test, y_train, y_test = train_test_split(
            dssv.X,
            dssv.y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return SplitLabeledTensor(
            train=LabeledTensor(X=X_train, y=y_train),
            test=LabeledTensor(X=X_test, y=y_test),
        )


class Field:
    """
    Dataframeのフィールド定義を管理するクラス
    dict型の定義を受け取り、カラム名やschemaなど柔軟な形式で返す。

    Properties:
        index: str | None
        label: list[str]
        exclude: list[str]
    """

    def __init__(
        self,
        definition: dict[str, PdType],
        index: str | None = None,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        self.definition = definition
        # Index: str | None
        self.index = index
        # Label: list[str]
        if label is None:
            self.label = []
        elif isinstance(label, str):
            self.label = [label]
        elif isinstance(label, list):
            self.label = label
        else:
            raise TypeError(f"不正なlabel => {label}")
        # Exclude: list[str]
        if exclude is None:
            self.exclude = []
        elif isinstance(exclude, str):
            self.exclude = [exclude]
        elif isinstance(exclude, list):
            self.exclude = exclude
        else:
            raise TypeError(f"不正なexclude => {exclude}")

    @property
    def col_names(self) -> list[str]:
        return list(self.definition.keys())

    @property
    def schema(self) -> DataFrameSchema:
        schema_dict = {
            column_name: Column(dtype) for column_name, dtype in self.definition.items()
        }
        return DataFrameSchema(schema_dict)
