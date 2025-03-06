from polars import Date, Float64, Int64, Schema

from common.schema import PLDF

# OHCLV: TypeAlias = pl.DataFrame
# OHCLV_SCHEMA = pl.Schema(
#     {
#         "Date": pl.Date,
#         "Open": pl.Float64,
#         "High": pl.Float64,
#         "Low": pl.Float64,
#         "Close": pl.Float64,
#         "Volume": pl.Int64,
#     }
# )


OHCLV = PLDF(
    schema=Schema(
        {
            "Date": Date,
            "Open": Float64,
            "High": Float64,
            "Low": Float64,
            "Close": Float64,
            "Volume": Int64,
        }
    ),
    origins=None,
)
