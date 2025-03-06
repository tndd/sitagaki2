from dataclasses import dataclass

from polars import DataFrame, Schema


@dataclass
class PLDF:
    schema: Schema
    origins: list["PLDF"] | None
    t: type[DataFrame] = DataFrame

    def columns(self):
        """
        カラム名のリストを取得する
        """
        pass
