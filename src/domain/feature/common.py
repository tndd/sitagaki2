from pandas import DataFrame
from pandera.api.pandas.types import PandasDtypeInputTypes as PanderaType

from infra.model.dataset import LabeledDataset


class OhlcvFeature(LabeledDataset):
    """
    ohlcvを元として生成される特徴量を表すクラス。

    Props:
        df:
            ohlcvと特徴量のカラムを持つデータフレーム。
        definition:
            dfのカラムの型定義。
            ただしここは特徴量の定義のみを含む。
            ohlcvの定義については含んではいない。
        label:
            dfのラベルカラム名。


    ohlcvを要素として持つ理由:
        バックテストで特徴量のパフォーマンスの検証を行うため。
        ohlcvが無いと、具体的な価格の推移を計算できないから。

    TODO: dfのカラム検討
        今のところdfのカラムはデフォルトでは特徴量の分しか持っていない。
        だが本当はohlcvのものもデフォルトで持たせて、
        必要に応じて特徴量のみに制限するという運用の方が適切ではないだろうか？

    MEMO: 設計方針
        dfにはohlcvのものも含める。
        ただし、特徴量のみのdfを返すメソッドを用意しておく。
        ohlcvのカラムは決まったものであるため、それの実現は容易だ。

        これによってdf生成メソッドのややこしさ問題を回避できるだろう。
        それを考えることなく、完成済みのdfを受け取るだけで話が済む。
    """

    def __init__(
        self,
        df: DataFrame,
        definition: dict[str, PanderaType],
        label: str,
    ) -> None:
        super().__init__(
            df=df,
            definition=definition,
            label=label,
            index="Date",
        )

    @property
    def feature_df(self) -> DataFrame:
        """
        特徴量のみのデータフレームを返す。
        学習済みのモデルに与えるための値として使う。
        """
        return self.df.loc[:, list(self.definition.keys())]
