from dataclasses import dataclass

from lightgbm import Dataset
from numpy import ndarray


@dataclass
class LabeledDataset:
    """
    教師ありデータセットのフォーマット
    (Supervised)

    X:
        学習データの特徴量を表す。
        大文字なのは、値が複数であることを表すため。

    y:
        目的変数である教師データを表す。
        小文字なのは、原則的には目的変数が単数であることを表すため。
        (yは複数形を取り得ることもあり得る)
    """

    X: ndarray
    y: ndarray

    def to_lgb(self) -> Dataset:
        return Dataset(self.X, self.y)


@dataclass
class LabeledDatasetSplit:
    """
    訓練と検証のために分割された、
    教師ありデータセットのフォーマット
    """

    train: LabeledDataset
    test: LabeledDataset

    def get_lgb_train_test(self) -> tuple[Dataset, Dataset]:
        lgb_train = self.train.to_lgb()
        lgb_test = Dataset(
            data=self.test.X,
            label=self.test.y,
            reference=lgb_train,
        )
        return lgb_train, lgb_test
