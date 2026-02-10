from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplitter:
    """
    Walk-forward CV with purging + embargo.

    We split on unique dates into contiguous test blocks.
    For each block:
      - Test set = events with date in the test block
      - Purge from train any event whose label window overlaps the test window
      - Embargo: additionally remove events within embargo_days after the test window

    Notes:
      - This is a daily-MVP implementation using calendar days.
      - Use a conservative label_horizon_days to cover trading-day horizon.
    """

    n_splits: int = 5
    label_horizon_days: int = 7  # approx for CAR[+1,+5]
    embargo_days: int = 5

    def split(self, dates: pd.Series) -> list[tuple[int, np.ndarray, np.ndarray, tuple[str, str]]]:
        """
        dates: pd.Series of event "formation" dates (trade_date_aligned), convertible to datetime
        returns list of (fold, train_mask, test_mask, (test_start, test_end))
        """
        dts = pd.to_datetime(dates).dt.normalize()
        unique_dates = np.array(sorted(dts.dt.date.unique()))
        if len(unique_dates) < self.n_splits * 5:
            # still works, but folds may be tiny
            pass

        blocks = np.array_split(unique_dates, self.n_splits)
        out = []

        for fold, test_dates in enumerate(blocks):
            if len(test_dates) == 0:
                continue

            test_start = pd.Timestamp(test_dates[0])
            test_end = pd.Timestamp(test_dates[-1])

            # test mask: date in test_dates
            test_mask = dts.dt.date.isin(test_dates).to_numpy()

            # Purge rule:
            # label window for an event at date t is [t+1, t+label_horizon_days]
            # It overlaps test window [test_start, test_end] if:
            #   (t + 1) <= test_end AND (t + label_horizon_days) >= test_start
            # Solve for t:
            #   t <= test_end - 1
            #   t >= test_start - label_horizon_days
            purge_lo = test_start - pd.Timedelta(days=self.label_horizon_days)
            purge_hi = test_end - pd.Timedelta(days=1)

            # Candidates that could overlap -> exclude from train
            overlap_mask = (dts >= purge_lo) & (dts <= purge_hi)

            # Embargo rule: exclude events in (test_end, test_end + embargo_days]
            embargo_hi = test_end + pd.Timedelta(days=self.embargo_days)
            embargo_mask = (dts > test_end) & (dts <= embargo_hi)

            # Base train: strictly before test_start (keeps walk-forward)
            base_train = dts < test_start

            train_mask = (base_train & ~overlap_mask & ~embargo_mask).to_numpy()

            # Skip folds with empty train or test (happens for earliest block)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            out.append(
                (
                    fold,
                    train_mask,
                    test_mask,
                    (str(test_start.date()), str(test_end.date())),
                )
            )

        return out
