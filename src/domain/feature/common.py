from pandas import DataFrame, concat
from pandera.api.pandas.types import PandasDtypeInputTypes as PdType

from domain.dataset.ohlcv import Ohlcv
from infra.model.dataset import Dataset


class OhlcvFeature(Dataset):
    """
    ohlcvを元として生成される特徴量を表すクラス。

    ohlcvを要素として持つ理由:
        バックテストで特徴量のパフォーマンスの検証を行うため。
        ohlcvが無いと、具体的な価格の推移を計算できないから。

    このクラスの前提条件:
        生成素材をohlcv"のみ"ということを前提としている。
        ohlcv以外も素材として特徴量を生成することは十分考えられるが、
        それはこのクラスの責務外とする。

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
            df=self._feature_df_source(ohlcv),
            # indexやlabelも特徴量についてのものを設定
            index=index,
            label=label,
            exclude=exclude,
        )

    @staticmethod
    def _feature_df_source(ohlcv: Ohlcv) -> DataFrame:
        """
        抽象メソッド。
        ohlcvを元に特徴量を生成する。
        ここで生成されたdfが特徴量のdfとなる。

        setterという命名について:
            実際は値のセットは行っておらず、生成部分までしかやってない。
            だが派生クラスから見た場合に役割をわかりやすくするための方便的な命名。
        """
        raise NotImplementedError("This method should be implemented by subclass.")

    @property
    def df_with_ohlcv(self) -> DataFrame:
        """
        自身の特徴量dfとohlcv.dfを結合して返す。
        結合しないとバックテストで使えない。
        """
        return concat(
            [self.ohlcv.df, self.df],
            axis=1,
            join="inner",
        )
