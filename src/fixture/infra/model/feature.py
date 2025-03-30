from pandas import DataFrame

from fixture.infra.model.dataset import DatasetImpl, factory_dataset_impl
from infra.model.dataset import Dataset
from infra.model.feature import Feature


class FeatureImpl(Feature):
    """
    特徴量を表すクラスのテスト用fixture
    """

    SCHEMA = {
        "Date": "INDEX",
        "f0": float,
        "f1": float,
        "f2": float,
        "f3": float,
        "f4": float,
    }

    def __init__(self, dataset: DatasetImpl) -> None:
        super().__init__(
            dataset,
            index="Date",
            label="f0",
        )

    @staticmethod
    def derive_df(dataset: Dataset) -> DataFrame:
        """
        渡されたdfのインデックスを保持し、
        SCHEMAに基づいて連続した値を持つdfを生成する。

        生成される値の例:
        ┌───────┬─────┬─────┬─────┬─────┬─────┐
        │ index │ f0  │ f1  │ f2  │ f3  │ f4  │
        ├───────┼─────┼─────┼─────┼─────┼─────┤
        │ a     │ 0.1 │ 1.1 │ 2.1 │ 3.1 │ 4.1 │
        │ b     │ 0.2 │ 1.2 │ 2.2 │ 3.2 │ 4.2 │
        │ c     │ 0.3 │ 1.3 │ 2.3 │ 3.3 │ 4.3 │
        │ ...   │ ... │ ... │ ... │ ... │ ... │
        └───────┴─────┴─────┴─────┴─────┴─────┘
        """
        data = {
            col: [0.1 + i + 0.1 * j for j in range(len(dataset.df.index))]
            for i, col in enumerate(FeatureImpl.SCHEMA.keys())
        }
        return DataFrame(data, index=dataset.df.index)


def factory_feature_impl() -> FeatureImpl:
    return FeatureImpl(factory_dataset_impl())
