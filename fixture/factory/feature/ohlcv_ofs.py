from feature.ohlcv_ofs.derive import derive_df_ohlcv_ofs
from feature.ohlcv_ofs.schema import OhlcvOfs
from fixture.factory.dataset.ohlcv import factory_ohlcv


def factory_ohlcv_ofs() -> OhlcvOfs:
    df = factory_ohlcv()
    return derive_df_ohlcv_ofs(df)


if __name__ == "__main__":
    df = factory_ohlcv_ofs()
    print(df)
