from polars import DataFrame, Schema


class PLDF:
    schema: Schema
    origins: list["PLDF"] | None = None

    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df

    @property
    def columns(self):
        """
        カラム名のリストを取得する
        """
        return self.schema.names()
