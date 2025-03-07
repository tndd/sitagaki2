from feature.repository import derive_df_ofs_ohclv
from feature.schema import OFS_OHCLV
from fixture.factory.dataset.ohlcv import factory_ohlcv


def factory_ofs_ohclv() -> OFS_OHCLV:
    df = factory_ohlcv()
    return derive_df_ofs_ohclv(df)


if __name__ == "__main__":
    df = factory_ofs_ohclv()
    print(df)
