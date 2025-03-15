from dataclasses import dataclass

from lightgbm import Booster, Dataset, early_stopping, train
from sklearn.model_selection import train_test_split

from domain.feature.closes.schema import ClosesN4

params_base = {
    "objective": "regression",  #   #  タスクに応じて変更（例: 'binary', 'multiclass'）
    "metric": "rmse",  #            #  評価指標（例: 'auc', 'logloss'）
    "boosting_type": "gbdt",  #     #  デフォルトの勾配ブースティング
    "num_leaves": 31,  #            #  木の複雑さを調整（大きいほど複雑）
    "learning_rate": 0.05,  #       #  学習率（小さくすると精度が上がるが遅くなる）
    "feature_fraction": 0.9,  #     #  各木で使う特徴量の割合（過学習防止）
    "bagging_fraction": 0.8,  #     #  データのサブサンプリング割合（過学習防止）
    "bagging_freq": 5,  #           #  バギングの頻度
    "verbose": -1,  #               #  ログを非表示
}


@dataclass
class DataForModel:
    train: Dataset
    valid: Dataset

    @classmethod
    def from_xy(cls, X, y, test_size=0.2, random_state=42):
        """
        注意: データは昇順に並んでるという前提
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=random_state
        )
        data_train = Dataset(X_train, label=y_train)
        data_valid = Dataset(X_test, label=y_test, reference=data_train)

        return cls(data_train, data_valid)


def train_model_lgbm_closes_n4(train_set: ClosesN4, valid_set: ClosesN4) -> Booster:
    # TODO: 入力はtrain_setのみというシンプルな形に
    return train(
        params_base,
        train_set.df,
        valid_sets=[valid_set.df],
        num_boost_round=500,
        callbacks=[early_stopping(50)],
    )
