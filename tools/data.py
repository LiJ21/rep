import os
import re
from datetime import datetime, timedelta
from typing import Any

import exchange_calendars as ec
import polars as pl


# maps an exclude_holiday key to its exchange_calendars calendar code; extend as needed
_HOLIDAY_CALENDARS = {"cme": "CMES"}


def expand_dates(dates, exclude_holiday="cme", str_result=True):
    parts = dates.split("-")
    start = datetime.strptime(parts[0], "%Y%m%d").date()
    end = datetime.strptime(parts[-1], "%Y%m%d").date()
    if end < start:
        raise ValueError(f"end date {parts[-1]} precedes start date {parts[0]}")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if exclude_holiday:
        cal = _HOLIDAY_CALENDARS.get(exclude_holiday.lower())
        if cal is None:
            raise ValueError(
                f"unsupported exclude_holiday {exclude_holiday!r}; supported: {sorted(_HOLIDAY_CALENDARS)}"
            )
        sessions = set(ec.get_calendar(cal).sessions_in_range(start, end).date)
        days = [d for d in days if d in sessions]

    return [d.isoformat() for d in days] if str_result else days


RAW_PATH = "/home/jli/projects/rep/data/databento_glbx_mdp3_mbo_full_day_parquet/data/databento_glbx_mdp3_mbo_full_day_parquet/{prod}M6_{d}_extra_(normal|stress)_{prod_s}_full_day.parquet"
class Raw:
    @classmethod
    def load_date(cls, d, prod, path=RAW_PATH, filters=[], cols=None) -> tuple[str | Any, LazyFrame]:
        p = path.format(
            prod=prod,
            prod_s=prod.lower(),
            d=d,
            dnd=d.replace("-", ""),
            dslash=d.replace("-", "/"),
        )

        d_dir, name_pat = os.path.split(p)
        regex = re.compile("^" + name_pat + "$")
        matched = [(f, m.group(1)) for f in os.listdir(d_dir) if (m := regex.match(f))]

        if len(matched) != 1:
            raise ValueError(
                f"expected exactly one file matching {name_pat!r} in {d_dir!r}, "
                f"matched {[w for _, w in matched]}"
            )

        fname, word = matched[0]
        print(f"matched word: {word}")
        p = os.path.join(d_dir, fname)

        df_scan = pl.scan_parquet(p)
        mask = pl.lit(True)
        for f in filters:
            mask = mask & f
        df = df_scan.filter(mask).select(cols if cols is not None else df_scan.columns)
        return word, df

    @classmethod
    def load_dates(cls, dates, prod, **kwargs):
        rdates = dates if isinstance(list) else expand_dates(dates)
        dfs = [cls.load_date(d, prod, **kwargs) for d in rdates]

        df = pl.concat([df for _, df in dfs])
        nature = {d: w for d, w in zip(rdates, [w for w, _ in dfs])}

        return df, nature
