from domain.feature.ohlcv_ofs import OhlcvOfs, derive_df_ohlcv_ofs
from fixture.domain.dataset.ohlcv import factory_ohlcv


def factory_ohlcv_ofs() -> OhlcvOfs:
    df = factory_ohlcv()
    return derive_df_ohlcv_ofs(df)


if __name__ == "__main__":
    df = factory_ohlcv_ofs()
    print(df)
