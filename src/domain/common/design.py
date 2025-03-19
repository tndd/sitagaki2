from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame as DataFramePD
from polars import DataFrame, Schema
from sklearn.model_selection import train_test_split


@dataclass
class LabeledDataset:
    """
    教師ありデータセットのフォーマット
    (Supervised)

    X:
        学習データの特徴量を表す。
        大文字なのは、値が複数であることを表すため。

    y:
        目的変数である教師データを表す。
        小文字なのは、原則的には目的変数が単数であることを表すため。
        (yは複数形を取り得ることもあり得る)
    """

    X: DataFramePD
    y: ndarray


@dataclass
class LabeledDatasetSplit:
    """
    訓練と検証のために分割された、
    教師ありデータセットのフォーマット
    """

    train: LabeledDataset
    test: LabeledDataset


class Pldf:
    """
    Polars dataframeを扱うための抽象クラス

    DFをそのまま使うと、単なるDataframe型となってしまい、
    データの把握が困難になるため。
    スキーマ定義と親の情報を持つ。
    """

    SCHEMA: Schema
    ORIGIN: list["Pldf"] | None = None

    def __init__(
        self,
        df: DataFrame,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """
        df: pl.DataFrame
            polars dataframeはここに格納される。

        label: list[str]
            教師データのラベル名を指定する。
            ラベルがない場合は、何も入れない。

        exclude: list[str]
            特徴量としては含めない項目を指定する。
            想定としては、Dateのような日付データなど。
        """
        self.df: DataFrame = df
        self.label: str | list[str] | None = label
        # excludeを常にリストに翻訳
        if exclude is None:
            self.exclude = []
        elif isinstance(exclude, str):
            self.exclude = [exclude]
        elif isinstance(exclude, (list, tuple, set)):
            self.exclude = list(exclude)
        else:
            raise TypeError(f"不正なexclude => {exclude}")

    @classmethod
    def get_col_names(cls) -> list[str]:
        """
        カラム名のリストを取得する
        """
        return cls.SCHEMA.names()

    def get_labeled_dataset(self) -> LabeledDataset:
        """
        指定されたlabelを教師データカラムに。
        そしてその他のカラムを学習データとして、
        教師あり学習トレーニング用のDatasetSVに加工して返す。
        """
        if self.label is None:
            # labels未定義状態で、この関数を呼んだ場合はエラーで落とす
            raise ValueError("There is no label in this schema.")
        elif isinstance(self.label, list):
            raise ValueError("WIP: Labelがlist型の動作は未定義。")
        else:
            return LabeledDataset(
                X=self.df.drop(self.label).to_pandas(),
                y=self.df[self.label].to_numpy(),
            )

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
