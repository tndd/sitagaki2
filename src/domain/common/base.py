from dataclasses import dataclass

from numpy import ndarray
from polars import DataFrame, Schema


@dataclass
class DatasetSv:
    """
    教師ありデータセットのフォーマット
    (Supervised)
    """

    train: ndarray
    label: ndarray


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
        self.label = label

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
                train=self.df.select(self.df.columns.exclude([self.label])).to_numpy(),
                label=self.df[self.label].to_numpy(),
            )
