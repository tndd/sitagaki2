from dataclasses import dataclass

from polars import DataFrame, Schema, Series


@dataclass
class DatasetSV:
    """
    教師ありデータセットのフォーマット
    (SuperVised)
    """

    train: DataFrame
    label: Series


@dataclass
class DatasetSVML:
    """
    複数ラベルの
    教師ありデータセットのフォーマット
    (SuperVised MultiLabel)
    """

    train: DataFrame
    labels: DataFrame


class Pldf:
    """
    Polars dataframeを扱うための抽象クラス

    DFをそのまま使うと、単なるDataframe型となってしまい、
    データの把握が困難になるため。
    スキーマ定義と親の情報を持つ。
    """

    SCHEMA: Schema
    ORIGIN: list["Pldf"] | None = None

    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df

    @classmethod
    def get_col_names(cls):
        """
        カラム名のリストを取得する
        """
        return cls.SCHEMA.names()

    def get_dataset_sv(self, label: str):
        """
        指定されたlabelを教師データカラムに。
        そしてその他のカラムを学習データとして、
        教師あり学習トレーニング用のDatasetSVに加工して返す。
        """
        return DatasetSV(
            train=self.df.select(self.df.columns.exclude([label])),
            label=self.df[label],
        )
