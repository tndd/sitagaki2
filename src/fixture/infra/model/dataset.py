from pandas import DataFrame, to_datetime

from infra.model.draft import Dataset


class DatasetImpl(Dataset):
    def __init__(
        self,
        df: DataFrame,
    ) -> None:
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


class DatasetImplV2(Dataset):
    def __init__(
        self,
        df: DataFrame,
    ) -> None:
        super().__init__(
            df,
            definition={
                "D0": float,
                "D1": float,
                "D2": float,
                "D3": float,
                "D4": float,
            },
            index="Date",
        )


class LabeledDatasetImpl(Dataset):
    def __init__(
        self,
        df: DataFrame,
    ) -> None:
        super().__init__(
            df,
            definition={
                "F0": float,
                "F1": float,
                "F2": float,
                "F3": float,
                "L": float,
            },
            index="Date",
            label="L",
        )


def factory_dataset_impl() -> DatasetImpl:
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
    return DatasetImpl(df)


def factory_dataset_impl_v2(
    label: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
) -> DatasetImplV2:
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
            "D0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "D1": [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
            "D2": [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
            "D3": [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3],
            "D4": [0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4],
        }
    )
    return DatasetImplV2(df)
