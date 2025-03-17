from datetime import datetime
from math import log

from domain.feature.common.const import SCALE_BP
from domain.feature.ohlcv_ofs.derive import derive_df_ohlcv_ofs
from domain.feature.ohlcv_ofs.schema import OhlcvOfs
from fixture.common.const import APX_ZERO
from fixture.factory.dataset.ohlcv import factory_ohlcv


def test_derive_df_ohlcv_ofs():
    ohlcv = factory_ohlcv()
    ohlcv_ofs = derive_df_ohlcv_ofs(ohlcv)
    assert isinstance(ohlcv_ofs, OhlcvOfs)
    # dfのスキーマとクラスのスキーマを比べる
    assert ohlcv_ofs.df.schema == OhlcvOfs.SCHEMA
    # derive前の4件から１つ減って3件となってるか?
    assert len(ohlcv_ofs.df) == 3
    # 2000-01-01は削除されてるから2000-01-02から始まってるか?
    assert ohlcv_ofs.df["Date"][0] == datetime(2000, 1, 2)
    # PrevCloseOfsの値動きの合計は0となってるか？
    assert ohlcv_ofs.df["PrevCloseOfs"].sum() == APX_ZERO
    # PrevVolumeOfsの値動きの合計は0となってるか？
    assert ohlcv_ofs.df["PrevVolumeOfs"].sum() == APX_ZERO
    # 2000-01-04のCloseは0となっているか？
    assert ohlcv_ofs.df["Date"][2] == datetime(2000, 1, 4)
    assert ohlcv_ofs.df["CloseOfs"][2] == APX_ZERO
    ### 値の計算 ###
    # 2000-01-02のPrevCloseOfs
    assert (
        log(ohlcv.df["Close"][1] / ohlcv.df["Close"][0]) * SCALE_BP
        == ohlcv_ofs.df["PrevCloseOfs"][0]
    )
    # 2000-01-03のHLCのOfs
    assert (
        log(ohlcv.df["High"][2] / ohlcv.df["Open"][2]) * SCALE_BP
        == ohlcv_ofs.df["HighOfs"][1]
    )
    assert (
        log(ohlcv.df["Low"][2] / ohlcv.df["Open"][2]) * SCALE_BP
        == ohlcv_ofs.df["LowOfs"][1]
    )
    assert (
        log(ohlcv.df["Close"][3] / ohlcv.df["Open"][3]) * SCALE_BP
        == ohlcv_ofs.df["CloseOfs"][2]
    )
    # 2000-01-04のVolumeOfs
    assert (
        log(ohlcv.df["Volume"][3] / ohlcv.df["Volume"][2]) * SCALE_BP
        == ohlcv_ofs.df["PrevVolumeOfs"][2]
    )
