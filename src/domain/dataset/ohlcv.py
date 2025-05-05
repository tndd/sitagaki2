from pandas import DataFrame

from infra.model.dataset import Dataset


class Ohlcv(Dataset):
    def __init__(self, df: DataFrame) -> None:
        """
        IndexはDateで固定
        """
        super().__init__(
            df,
            definition={
                "Open": float,
                "High": float,
                "Low": float,
                "Close": float,
                "Volume": int,
            },
            index="Date",
        )
