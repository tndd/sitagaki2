from dataset.ohlcv.schema import OHLCV
from feature.closes.schema import CLOSES_N4, CLOSES_N4_PLDF
from feature.closes.service import derive_closes


# TODO: テスト実装
def derive_closes_n4(df: OHLCV) -> CLOSES_N4:
    return derive_closes(df, 4).select(CLOSES_N4_PLDF.columns)
