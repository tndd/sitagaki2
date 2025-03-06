from typing import TypeAlias

from polars import DataFrame

OHCLV: TypeAlias = DataFrame
"""
date:   dt
open:   f64
high:   f64
low:    f64
close:  f64
volume: i64
"""
