from datetime import datetime
from math import log

from common.const import SCALE_BP
from feature.ofs_ohlcv.derive import derive_df_ofs_ohlcv
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
    # 2000-01-04のCloseは0となっているか？
    assert ofs_ohlcv["Date"][2] == datetime(2000, 1, 4)
    assert ofs_ohlcv["CloseOfs"][2] == APX_ZERO
    ### 値の計算 ###
    # 2000-01-02のPrevCloseOfs
    assert (
        log(df["Close"][1] / df["Close"][0]) * SCALE_BP == ofs_ohlcv["PrevCloseOfs"][0]
    )
    # 2000-01-03のHLCのOfs
    assert log(df["High"][2] / df["Open"][2]) * SCALE_BP == ofs_ohlcv["HighOfs"][1]
    assert log(df["Low"][2] / df["Open"][2]) * SCALE_BP == ofs_ohlcv["LowOfs"][1]
    assert log(df["Close"][3] / df["Open"][3]) * SCALE_BP == ofs_ohlcv["CloseOfs"][2]
    # 2000-01-04のVolumeOfs
    assert (
        log(df["Volume"][3] / df["Volume"][2]) * SCALE_BP
        == ofs_ohlcv["PrevVolumeOfs"][2]
    )
