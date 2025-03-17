from dataclasses import dataclass

from numpy import ndarray
from polars import DataFrame, Schema
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSv:
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

    X: ndarray
    y: ndarray


@dataclass
class DatasetSvSplited:
    """
    訓練と検証のために分割された、
    教師ありデータセットのフォーマット
    """

    train: DatasetSv
    test: DatasetSv


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
    ) -> None:
        """
        df:
            polars dataframeはここに格納される。

        label:
            教師データのラベル名を指定する。
            ラベルがない場合は、何も入れない。
        """
        self.df: DataFrame = df
        self.label: str | list[str] | None = label

    @classmethod
    def get_col_names(cls) -> list[str]:
        """
        カラム名のリストを取得する
        """
        return cls.SCHEMA.names()

    def get_dataset_sv(self) -> DatasetSv:
        """
        指定されたlabelを教師データカラムに。
        そしてその他のカラムを学習データとして、
        教師あり学習トレーニング用のDatasetSVに加工して返す。
        """
        if self.label is None:
            # labels未定義状態でget_dataset_sv()を使った場合はエラーで落とす
            raise ValueError("There is no label in this schema.")
        elif isinstance(self.label, list):
            raise ValueError("WIP: Labelがlist型の動作は未定義。")
        else:
            return DatasetSv(
                X=self.df.select(self.df.columns.exclude([self.label])).to_numpy(),
                y=self.df[self.label].to_numpy(),
            )

    def get_dataset_sv_splited(
        self,
        test_size: int = 0.2,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> DatasetSvSplited:
        """
        訓練用とテスト用にデータが分割されたDatasetSvを返す

        原則的には時系列を考慮し、シャッフルはしない。
        テストサイズは20%。
        """
        dssv = self.get_dataset_sv()
        # 時系列を考慮したデータ分割
        X_train, X_test, y_train, y_test = train_test_split(
            dssv.X,
            dssv.y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return DatasetSvSplited(
            train=DatasetSv(X=X_train, y=y_train),
            test=DatasetSv(X=X_test, y=y_test),
        )
