from datetime import datetime

from feature.derive import derive_df_ofs_ohlcv
from fixture.common.const import APX_ZERO
from fixture.factory.dataset.ohlcv import factory_ohlcv


def test_derive_df_ofs_ohlcv():
    df = factory_ohlcv()
    ofs_ohlcv = derive_df_ofs_ohlcv(df)
    # derive前の4件から１つ減って3件となってるか?
    assert len(ofs_ohlcv) == 3
    # 2000-01-01は削除されてるから2000-01-02から始まってるか?
    assert ofs_ohlcv["Date"][0] == datetime(2000, 1, 2)
    # PrevCloseOfsの値動きの合計は0となってるか？
    assert ofs_ohlcv["PrevCloseOfs"].sum() == APX_ZERO
    # PrevVolumeOfsの値動きの合計は0となってるか？
    assert ofs_ohlcv["PrevVolumeOfs"].sum() == APX_ZERO
