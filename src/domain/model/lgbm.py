from lightgbm import Booster, early_stopping, train

from domain.feature.closes.schema import ClosesN4


def train_model_lgbm_closes_n4(dataset: ClosesN4) -> Booster:
    params = {
        "objective": "regression",  #   #  例: 'binary', 'multiclass'
        "metric": "rmse",  #            #  評価指標（例: 'auc', 'logloss'）
        "boosting_type": "gbdt",  #     #  デフォルトの勾配ブースティング
        "num_leaves": 31,  #            #  木の複雑さを調整（大きいほど複雑）
        "learning_rate": 0.05,  #       #  学習率（小さくすると精度が上がるが遅くなる）
        "feature_fraction": 0.9,  #     #  各木で使う特徴量の割合（過学習防止）
        "bagging_fraction": 0.8,  #     #  データのサブサンプリング割合（過学習防止）
        "bagging_freq": 5,  #           #  バギングの頻度
        "verbose": -1,  #               #  ログを非表示
    }
    lds = dataset.get_labeled_dataset_split()
    lgb_train, lgb_test = lds.get_lgb_train_test()
    return train(
        params,
        lgb_train,
        valid_sets=[lgb_test],
        num_boost_round=500,
        callbacks=[early_stopping(50)],
    )
