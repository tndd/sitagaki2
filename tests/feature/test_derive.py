from feature.derive import derive_df_ofs_ohlcv
from fixture.factory.dataset.ohlcv import factory_ohlcv


def test_derive_df_ofs_ohlcv():
    df = factory_ohlcv()
    ofs_ohlcv = derive_df_ofs_ohlcv(df)
    assert len(ofs_ohlcv) == 3
    # TODO: 続き
