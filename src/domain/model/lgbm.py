from lightgbm import (
    Booster,
    early_stopping,
    train,
)

from domain.feature.common import OhlcvFeature
from infra.model.tensor import convert_labeled_dataset_to_split_tensor


def train_lgbm_with_ohlcv_feature(
    dataset: OhlcvFeature,
    init_model: Booster | None = None,
) -> Booster:
    params = {
        "objective": "regression",  # 回帰問題として解く
        "metric": "rmse",  # 評価指標
        "boosting_type": "gbdt",  # 勾配ブースティング
        "num_leaves": 31,  # 木の複雑さ
        "learning_rate": 0.05,  # 学習率
        "feature_fraction": 0.9,  # 特徴選択の割合
        "bagging_fraction": 0.8,  # データのサブサンプリング割合
        "bagging_freq": 5,  # バギングの頻度
        "verbose": -1,  # ログを非表示
    }
    split_labeled_tensor = convert_labeled_dataset_to_split_tensor(dataset)
    lgb_train, lgb_test = split_labeled_tensor.get_lgb_train_test()
    return train(
        params,
        lgb_train,
        valid_sets=[lgb_test],
        num_boost_round=500,  # ラウンド数
        callbacks=[early_stopping(50)],  # 早期停止の監視期間
        init_model=init_model,  # 既存モデルを指定
    )
