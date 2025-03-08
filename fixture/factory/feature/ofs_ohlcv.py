from feature.derive import derive_df_ofs_ohlcv
from feature.schema import OFS_OHLCV
from fixture.factory.dataset.ohlcv import factory_ohlcv


def factory_ofs_ohlcv() -> OFS_OHLCV:
    df = factory_ohlcv()
    return derive_df_ofs_ohlcv(df)


if __name__ == "__main__":
    df = factory_ofs_ohlcv()
    print(df)
