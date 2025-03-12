from dataclasses import dataclass

from polars import DataFrame, Schema


@dataclass
class PLDF:
    df: DataFrame
    schema: Schema
    origins: list["PLDF"] | None = None

    @property
    def columns(self):
        """
        カラム名のリストを取得する
        """
        return self.schema.names()
