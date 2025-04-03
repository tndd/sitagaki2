from lightgbm import (
    Booster,
    early_stopping,
    train,
)

from domain.feature.common import OhlcvFeature


# --- モデル訓練関数 ---
def train_model_lgbm_ohlcv_feature(dataset: OhlcvFeature) -> Booster:
    """
    lgbmモデルをohlcv_featureデータセットを使って訓練する
    """
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
    slt = dataset.get_split_labeled_tensor()
    lgb_train, lgb_test = slt.get_lgb_train_test()
    return train(
        params,
        lgb_train,
        valid_sets=[lgb_test],
        num_boost_round=500,  # ラウンド数
        callbacks=[early_stopping(50)],  # 早期停止の監視期間
    )
