from pandas import DataFrame
from pandera import Column, DataFrameSchema
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType
from sklearn.model_selection import train_test_split

from infra.model.labeled_dataset import LabeledDataset, LabeledDatasetSplit


class DataSchema:
    """
    dataframeを扱うための抽象クラス
    スキーマ定義と親の情報を持つ。

    SCHEMA:
        indexについて:
            indexはスキーマ情報には含めない。
    """

    SCHEMA: dict[str, PdType]
    ORIGIN: list["DataSchema"] | None = None

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

        === Optional ===
            index: str | None
                インデックス名を指定する。
                インデックスがない場合は、Noneを指定する。

            label: list[str] | str | None
                教師データのラベル名を指定する。
                ラベルがない場合は、何も入れない。

            exclude: list[str] | str | None
                特徴量としては含めない項目を指定する。
                想定としては、Dateのような日付データなど。

        === クラス変数の焼き直し ===
            schema: dict[str, Column]
                スキーマを定義する。

            col_names: list[str]
                カラム名のリストを取得する。

        注意:
            labelとexcludeの入力型:
                入力の段階では、str, list, Noneの3つを取り得るが、
                内部的には一貫的にlistとして扱う。

        """
        # cls変数の代入
        self.schema: DataFrameSchema = self.__class__.get_df_schema()
        self.col_names: list[str] = self.__class__.get_col_names()
        # スキーマの定義と検証
        self.schema.validate(df)
        # 検証されたDataFrameを受け入れ
        self.df: DataFrame = df
        # Index
        if index is None:
            self.index = index
        elif isinstance(index, str):
            self.df = self.df.set_index(index)
        else:
            raise TypeError(f"不正なindex => {index}")
        # Label
        if label is None:
            self.label = []
        elif isinstance(label, str):
            self.label = [label]
        elif isinstance(label, list):
            self.label = label
        else:
            raise TypeError(f"不正なlabel => {label}")
        # Exclude
        if exclude is None:
            self.exclude = []
        elif isinstance(exclude, str):
            self.exclude = [exclude]
        elif isinstance(exclude, list):
            self.exclude = exclude
        else:
            raise TypeError(f"不正なexclude => {exclude}")

    @classmethod
    def get_col_names(cls) -> list[str]:
        """
        カラム名のリストを取得する
        """
        return list(cls.SCHEMA.keys())

    @classmethod
    def get_df_schema(cls) -> DataFrameSchema:
        """
        DataFrameSchemaを返す
        """
        schema_dict = {
            column_name: Column(dtype) for column_name, dtype in cls.SCHEMA.items()
        }
        return DataFrameSchema(schema_dict)

    def get_labeled_dataset(self) -> LabeledDataset:
        """
        指定されたlabelを教師データカラムに。
        そしてその他のカラムを学習データとして、
        教師あり学習トレーニング用のDatasetSVに加工して返す。
        """
        if len(self.label) == 0:
            # labels未定義状態で、この関数を呼んだ場合はエラーで落とす
            raise ValueError("There is no label in this schema.")
        elif len(self.label) == 1:
            # ラベルが１次元の場合
            return LabeledDataset(
                X=self.df.drop(columns=self.label + self.exclude).to_numpy(),
                y=self.df[self.label]
                .to_numpy()
                .ravel(),  # 1dラベルと確定しているので、ravelで警告を抑制
            )
        else:
            # 多次元ラベルの場合
            raise ValueError("WIP: Labelが複数の場合の動作は未定義")

    def get_labeled_dataset_split(
        self,
        test_size: int = 0.2,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> LabeledDatasetSplit:
        """
        訓練用とテスト用にデータが分割されたDatasetSvを返す

        原則的には時系列を考慮し、シャッフルはしない。
        テストサイズは20%。
        """
        dssv = self.get_labeled_dataset()
        # 時系列を考慮したデータ分割
        X_train, X_test, y_train, y_test = train_test_split(
            dssv.X,
            dssv.y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return LabeledDatasetSplit(
            train=LabeledDataset(X=X_train, y=y_train),
            test=LabeledDataset(X=X_test, y=y_test),
        )
