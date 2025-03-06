from dataclasses import dataclass

from polars import Schema


@dataclass
class PLDF:
    schema: Schema
    origins: list["PLDF"] | None

    def columns(self):
        """
        カラム名のリストを取得する
        """
        pass
