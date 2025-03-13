from dataset.ohlcv.schema import Ohlcv
from feature.closes.schema import ClosesN4
from feature.closes.service import derive_closes


def derive_closes_n4(ohlcv: Ohlcv) -> ClosesN4:
    return ClosesN4(derive_closes(ohlcv, 4).select(ClosesN4.get_col_names()))
