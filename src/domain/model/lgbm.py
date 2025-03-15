from lightgbm import Booster, early_stopping, train

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


def train_model_lgbm_closes_n4(train_set: ClosesN4, valid_set: ClosesN4) -> Booster:
    # TODO: 入力はtrain_setのみというシンプルな形に
    return train(
        params_base,
        train_set.df,
        valid_sets=[valid_set.df],
        num_boost_round=500,
        callbacks=[early_stopping(50)],
    )
