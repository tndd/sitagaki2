from pandas import DataFrame, to_datetime

from infra.model.dataset import Dataset


class DatasetImpl(Dataset):
    SCHEMA = {
        "Date": "INDEX",
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": int,
    }

    def __init__(
        self,
        df: DataFrame,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """
        IndexはDateで固定
        """
        super().__init__(
            df,
            index="Date",
            label=label,
            exclude=exclude,
        )


def factory_dataset_impl(
    label: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
) -> DatasetImpl:
    df = DataFrame(
        {
            "Date": to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2021-01-07",
                    "2021-01-08",
                    "2021-01-09",
                ]
            ),
            "Open": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
            "High": [110.0, 210.0, 310.0, 410.0, 510.0, 610.0, 710.0, 810.0, 910.0],
            "Low": [90.0, 190.0, 290.0, 390.0, 490.0, 590.0, 690.0, 790.0, 890.0],
            "Close": [105.0, 205.0, 305.0, 405.0, 505.0, 605.0, 705.0, 805.0, 905.0],
            "Volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        }
    )
    return DatasetImpl(
        df=df,
        label=label,
        exclude=exclude,
    )
