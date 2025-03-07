from feature.repository import derive_df_ofs_ohclv
from feature.schema import OFS_OHCLV
from fixture.factory.dataset.ohclv import factory_ohclv


def factory_ofs_ohclv() -> OFS_OHCLV:
    df = factory_ohclv()
    return derive_df_ofs_ohclv(df)


if __name__ == "__main__":
    df = factory_ofs_ohclv()
    print(df)
