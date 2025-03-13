from polars import DataFrame, Schema


class Pldf:
    schema: Schema
    origins: list["Pldf"] | None = None

    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df

    @classmethod
    def get_col_names(cls):
        """
        カラム名のリストを取得する
        """
        return cls.schema.names()
