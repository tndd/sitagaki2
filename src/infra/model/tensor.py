from dataclasses import dataclass

from lightgbm import Dataset
from numpy import ndarray
from sklearn.model_selection import train_test_split

from infra.model.draft import LabeledDataset


@dataclass
class LabeledTensor:
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
class SplitLabeledTensor:
    """
    訓練と検証のために分割された、
    教師ありデータセットのフォーマット
    """

    train: LabeledTensor
    test: LabeledTensor

    def get_lgb_train_test(self) -> tuple[Dataset, Dataset]:
        lgb_train = self.train.to_lgb()
        lgb_test = Dataset(
            data=self.test.X,
            label=self.test.y,
            reference=lgb_train,
        )
        return lgb_train, lgb_test


def convert_labeled_dataset_to_tensor(dataset: LabeledDataset) -> LabeledTensor:
    """
    教師ありデータのtensorに変換して返す
    """
    return LabeledTensor(
        X=dataset.df.drop(columns=dataset.non_label_columns).to_numpy(),
        y=dataset.df[dataset.label]
        .to_numpy()
        .ravel(),  # 1dラベルと確定しているので、ravelで警告を抑制
    )


def convert_labeled_dataset_to_split_tensor(
    dataset: LabeledDataset,
    test_size: int = 0.2,
    shuffle: bool = False,
    random_state: int = 42,
) -> SplitLabeledTensor:
    """
    訓練用とテスト用に分割された、教師ありtensorを返す

    基本設定:
        テストサイズは20%。
        時系列データが渡されることを考慮し、デフォルトではシャッフルはしない。
    """
    dssv = convert_labeled_dataset_to_tensor(dataset)
    # 時系列を考慮したデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        dssv.X,
        dssv.y,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    return SplitLabeledTensor(
        train=LabeledTensor(X=X_train, y=y_train),
        test=LabeledTensor(X=X_test, y=y_test),
    )
