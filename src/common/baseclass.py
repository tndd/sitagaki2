from polars import DataFrame, Schema


class PLDF:
    schema: Schema
    origins: list["PLDF"] | None = None

    def __init__(self, df: DataFrame) -> None:
        self.df: DataFrame = df

    @classmethod
    def get_col_names(cls):
        """
        カラム名のリストを取得する
        """
        return cls.schema.names()
