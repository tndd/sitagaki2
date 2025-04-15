from pandas import DataFrame, concat
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType

from domain.dataset.ohlcv import Ohlcv
from infra.model.dataset import Dataset

SCALE_BP = 10000


class OhlcvFeature(Dataset):
    """
    ohlcvを元として生成される特徴量を表すクラス

    ohlcvを要素として持つことで、
    バックテストで特徴量のパフォーマンスの検証が可能となる。

    Props:
        ohlcv: Ohlcv
        df: DataFrame
    ClsProps:
        SCHEMA: dict[str, PdType]
    """

    SCHEMA: dict[str, PdType]

    def __init__(
        self,
        ohlcv: Ohlcv,
        index: str = "Date",
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """
        ohlcvを受け取り、特徴量を生成する。
        """
        self.ohlcv = ohlcv
        super().__init__(
            # OhlcvFeatureのdfは特徴量のみで構成される
            df=self._derive_df(ohlcv),
            # indexやlabelも特徴量についてのものを設定
            index=index,
            label=label,
            exclude=exclude,
        )

    def _derive_df(self, ohlcv: Ohlcv) -> DataFrame:
        """
        抽象メソッド。
        ohlcvを元に特徴量を生成する。
        """
        raise NotImplementedError("This method should be implemented by subclass.")

    @property
    def df_merged(self) -> DataFrame:
        """
        自身の特徴量dfとohlcv.dfを結合して返す。
        結合しないとバックテストで使えない。
        """
        return concat(
            [self.ohlcv.df, self.df],
            axis=1,
            join="inner",
        )
