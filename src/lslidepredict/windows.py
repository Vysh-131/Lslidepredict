"""Dry-season acquisition windows around a landslide event.

NE India is reliably cloud-free only in the dry season (Nov - mid Feb, often into mid March),
so each temporal sample is drawn from one dry season. Y is always the calendar year of the event.

Case A - event Mar 1 .. Dec 31 (monsoon cycle of year Y):
    T0 (erosion baseline): Nov 1 (Y-2) -> Feb 15 (Y-1)
    T1 (pre-slide):        Nov 1 (Y-1) -> Feb 15 (Y)
    T2 (post-slide):       Nov 1 (Y)   -> Feb 15 (Y+1)      (start clamped after Nov/Dec events)

Case B - event Jan 1 .. Feb 29 (slide in the middle of a dry season), the winter is split:
    T0 (erosion baseline): Nov 1 (Y-2) -> Feb 15 (Y-1)
    T1 (pre-slide):        Nov 1 (Y-1) -> event - 1 day
    T2 (post-slide):       event + 1 day -> Mar 15 (Y)
        fallback:          Nov 1 (Y) -> Feb 15 (Y+1)  - used if the split window is shorter than
                           MIN_WINDOW_DAYS, or at mining time if it has no clear scene.

Every window is finally passed through a causal clamp (pre-event windows end before the
event, post-event windows start after it), so no window can ever contain the event date.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

MIN_WINDOW_DAYS = 15
ONE_DAY = timedelta(days=1)


@dataclass(frozen=True)
class Window:
    name: str
    role: str
    start: date
    end: date
    clamped: bool = False
    note: str = ""
    fallback: "Window | None" = None   # tried by the miner if this window yields no clear scene

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def valid(self) -> bool:
        return self.days >= MIN_WINDOW_DAYS

    @property
    def stac_datetime(self) -> str:
        return f"{self.start.isoformat()}T00:00:00Z/{self.end.isoformat()}T23:59:59Z"

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


def _clamp(w: Window, event: date, post: bool) -> Window:
    if post and w.start <= event:
        start = event + ONE_DAY
        return replace(w, start=start, clamped=True,
                       note=f"start clamped to {start} (day after event)")
    if not post and w.end >= event:
        end = event - ONE_DAY
        return replace(w, end=end, clamped=True, note=f"end clamped to {end} (day before event)")
    return w


def seasonal_windows(event: date | pd.Timestamp) -> list[Window]:
    ev = pd.Timestamp(event).date()
    y = ev.year
    t0 = Window("T0", "erosion baseline", date(y - 2, 11, 1), date(y - 1, 2, 15))

    if ev.month >= 3:  # Case A
        t1 = Window("T1", "pre-slide", date(y - 1, 11, 1), date(y, 2, 15))
        t2 = Window("T2", "post-slide", date(y, 11, 1), date(y + 1, 2, 15))
    else:              # Case B: split the current winter around the event
        next_winter = Window("T2", "post-slide", date(y, 11, 1), date(y + 1, 2, 15),
                             note="fallback: next winter")
        t1 = Window("T1", "pre-slide", date(y - 1, 11, 1), ev - ONE_DAY,
                    note="split winter (before event)")
        t2 = Window("T2", "post-slide", ev + ONE_DAY, date(y, 3, 15),
                    note="split winter (after event)", fallback=next_winter)
        if not t2.valid:
            t2 = replace(next_winter, note=f"split window < {MIN_WINDOW_DAYS} days; next winter")
        if not t1.valid:  # unreachable for Jan-Feb with a Nov 1 start, kept as a guard
            t1 = Window("T1", "pre-slide", date(y - 2, 11, 1), date(y - 1, 2, 15),
                        note=f"split window < {MIN_WINDOW_DAYS} days; previous winter")

    return [_clamp(t0, ev, post=False), _clamp(t1, ev, post=False), _clamp(t2, ev, post=True)]
