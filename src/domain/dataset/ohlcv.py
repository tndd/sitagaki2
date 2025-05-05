from pandas import DataFrame

from infra.model.dataset import Dataset

OHLCV_DEFINITION = {
    "Open": float,
    "High": float,
    "Low": float,
    "Close": float,
    "Volume": int,
}
OHLCV_INDEX = "Date"


class Ohlcv(Dataset):
    def __init__(self, df: DataFrame) -> None:
        super().__init__(
            df,
            definition=OHLCV_DEFINITION,
            index=OHLCV_INDEX,
        )
