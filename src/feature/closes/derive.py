from dataset.ohlcv.schema import Ohlcv
from feature.closes.schema import CLOSES_N4, CLOSES_N4_PLDF
from feature.closes.service import derive_closes


def derive_closes_n4(df: Ohlcv) -> CLOSES_N4:
    return derive_closes(df, 4).select(CLOSES_N4_PLDF.get_col_names)
