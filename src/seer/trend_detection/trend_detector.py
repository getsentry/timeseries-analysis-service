"""
Trend Detection Logic:

- Get the most recent change point (as long as its 5 datapoints from the left and 1 from the right)
    - otherwise if the change points are on the edges use midpoint analysis
- Calculate the trend percentage before/after the breakpoint
- Use Welch’s T-test to determine if there is a significant difference in means
    - Welch’s t-test: finds significant in means between two sample groups with unequal variances and different sample sizes
- If p-value > 0.01 and trend percentage > 5%, then the trend is surfaced

"""

import datetime
from typing import List, Literal, Mapping, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from seer.trend_detection.detectors.cusum_detection import CUSUMChangePoint, CUSUMDetector


class SnubaMetadata(TypedDict):
    count: float


# timestamp,
SnubaTSEntry = Tuple[int, Tuple[SnubaMetadata]]


class BreakpointTransaction(BaseModel):
    data: List[SnubaTSEntry]
    request_start: int
    request_end: int
    data_start: int
    data_end: int

    @field_validator("request_start", "request_end", "data_start", "data_end", mode="before")
    @classmethod
    def validate_ints(cls, v):
        return round(v)

    @validator('data', pre=True, each_item=True)
    @classmethod
    def validate_data(cls, v):
        new_data = []
        for timestamp, metadata_list in v:
            if len(metadata_list) > 1:
                aggregated_count = sum(item['count'] for item in metadata_list)
                new_metadata = {'count': aggregated_count}
                new_data.append((timestamp, [new_metadata]))
            else:
                new_data.append((timestamp, metadata_list))
        return new_data


class BreakpointRequest(BaseModel):
    data: Mapping[str, BreakpointTransaction]
    sort: str = ""
    allow_midpoint: str = "1"
    validate_tail_hours: int = 0
    trend_percentage: float = Field(default=0.1, alias="trend_percentage()")
    min_change: float = Field(default=0.0, alias="min_change()")


class BreakpointEntry(BaseModel):
    project: str
    # For legacy reasons, the group name is always
    # transaction even when working with functions.
    transaction: str
    aggregate_range_1: float
    aggregate_range_2: float
    unweighted_t_value: float
    unweighted_p_value: float
    trend_percentage: float
    absolute_percentage_change: float
    trend_difference: float
    breakpoint: int
    request_start: int
    request_end: int
    data_start: int
    data_end: int
    change: Union[Literal["improvement"], Literal["regression"]]


class BreakpointResponse(BaseModel):
    data: List[BreakpointEntry]


def find_changepoint(
    change_points: List[CUSUMChangePoint],
    timestamps: List[int],
    req_start: int,
    req_end: int,
    allow_midpoint: bool,
) -> int | None:
    # if breakpoints are detected, get most recent changepoint
    if change_points:
        change_point = int(datetime.datetime.timestamp(change_points[-1].start_time))
        change_index = timestamps.index(change_point)
        if change_index > 5:
            return change_point

    # check the midpoint boolean - don't get midpoint of the request period if this boolean is false, midpoint should only be used for trends
    if not allow_midpoint:
        return None

    return (req_start + req_end) // 2


def find_trends(
    txns_data: Mapping[str, BreakpointTransaction],
    sort_function: str,
    allow_midpoint: bool,
    min_pct_change: float,
    min_change: float,
    validate_tail_hours: int,
    pval=0.01,
) -> List[Tuple[float, BreakpointEntry]]:
    trend_percentage_list: List[Tuple[float, BreakpointEntry]] = []

    txn: BreakpointTransaction
    # defined outside for loop so error won't throw for empty data
    for txn_name, txn in txns_data.items():
        # data without zero-filling
        timestamps: List[int] = []
        metrics: List[float] = []

        # get all the non-zero data
        ts_data = txn.data
        ts_entry: SnubaTSEntry

        for timestamp, (metadata,) in txn.data:
            metric = metadata["count"]

            if metric != 0:
                timestamps.append(timestamp)
                metrics.append(metric)

        # snuba query limit was hit, and we won't have complete data for this transaction so disregard this txn_name
        if None in metrics:
            continue

        # extract all zero filled data
        timestamps_zero_filled: List[int] = [ts_data[x][0] for x in range(len(ts_data))]
        metrics_zero_filled: List[float] = [ts_data[x][1][0]["count"] for x in range(len(ts_data))]

        req_start = txn.request_start
        req_end = txn.request_end

        # don't include transaction if there are less than three datapoints in non zero data OR
        # don't include transaction if there is no more data within request time period
        if len(metrics) < 3 or req_start > timestamps[-1]:
            continue

        try:
            # grab the index of the request start time
            next(i for i, v in enumerate(timestamps) if v > req_start)
        except StopIteration:
            # After removing the zerofilled entries, it's possible that all
            # timestamps fall before the request start. When this happens, there
            # is no trend to be found.
            continue

        # convert to pandas timestamps for magnitude compare method in cusum detection
        timestamps_pandas: List[pd.Timestamp] = [
            pd.Timestamp(datetime.datetime.fromtimestamp(x)) for x in timestamps
        ]
        timestamps_zerofilled_pandas: List[pd.Timestamp] = [
            pd.Timestamp(datetime.datetime.fromtimestamp(x)) for x in timestamps_zero_filled
        ]

        timeseries = pd.DataFrame({"time": timestamps_pandas, "y": metrics})

        timeseries_zerofilled = pd.DataFrame(
            {"time": timestamps_zerofilled_pandas, "y": metrics_zero_filled}
        )

        change_points = CUSUMDetector(timeseries, timeseries_zerofilled).detector()
        change_points.sort(key=lambda x: x.start_time)

        change_point = find_changepoint(
            change_points, timestamps, req_start, req_end, allow_midpoint
        )
        if change_point is None:
            continue

        first_half = [
            metrics[i]
            for i in range(len(metrics))
            if timestamps[i] < change_point and timestamps[i] >= req_start
        ]
        second_half = [
            metrics[i]
            for i in range(len(metrics))
            if timestamps[i] >= change_point and timestamps[i] <= req_end
        ]

        # if either of the halves don't have any data to compare to then move on to the next txn_name
        if len(first_half) == 0 or len(second_half) == 0:
            continue

        mu0 = np.average(first_half)
        mu1 = np.average(second_half)

        # calculate t-value between both groups
        scipy_t_test = scipy.stats.ttest_ind(first_half, second_half, equal_var=False)

        if mu0 == 0:
            trend_percentage = mu1
        else:
            trend_percentage = mu1 / mu0

        txn_names = txn_name.split(",")

        entry = BreakpointEntry(
            project=txn_names[0],
            transaction=txn_names[1],
            aggregate_range_1=float(mu0),
            aggregate_range_2=float(mu1),
            unweighted_t_value=scipy_t_test.statistic,
            unweighted_p_value=round(scipy_t_test.pvalue, 10),
            trend_percentage=float(trend_percentage),
            absolute_percentage_change=abs(float(trend_percentage)),
            trend_difference=float(mu1 - mu0),
            breakpoint=change_point,
            request_start=req_start,
            request_end=req_end,
            data_start=int(txn.data_start),
            data_end=int(txn.data_end),
            change="regression",
        )

        # TREND LOGIC:
        #  1. p-value of t-test is less than passed in threshold (default = 0.01)
        #  2. trend percentage is greater than passed in threshold (default = 10%)
        #  3. last validate_tail_hours hours are also greater than threshold

        if validate_tail_hours > 0:
            validation_start = req_end - validate_tail_hours * 60 * 60

            # Filter out the data based on validate_tail_hours
            validation_data = [
                metric
                for timestamp, metric in zip(timestamps, metrics)
                if max(validation_start, change_point) <= timestamp <= req_end
            ]

            # Calculate the trend percentage and change for the last validate_tail_hours
            mu_validation = np.average(validation_data)
            trend_percentage_validation = mu_validation / mu0 if mu0 != 0 else mu_validation
            trend_change_validation = mu_validation - mu0
        else:
            trend_percentage_validation = None
            trend_change_validation = None

        # most improved - get only negatively significant trending txns
        if (
            (sort_function == "trend_percentage()" or sort_function == "")
            and mu1 + min_change <= mu0
            and scipy_t_test.pvalue < pval
            and abs(trend_percentage - 1) > min_pct_change
            and (
                trend_percentage_validation is None
                or abs(trend_percentage_validation - 1) > min_pct_change
            )
            and (trend_change_validation is None or abs(trend_change_validation) > min_change)
        ):
            entry.change = "improvement"
            trend_percentage_list.append((float(trend_percentage), entry))

        # if most regressed - get only positively significant txns
        elif (
            (sort_function == "-trend_percentage()" or sort_function == "")
            and mu0 + min_change <= mu1
            and scipy_t_test.pvalue < pval
            and trend_percentage - 1 > min_pct_change
            and (
                trend_percentage_validation is None
                or trend_percentage_validation - 1 > min_pct_change
            )
            and (trend_change_validation is None or trend_change_validation > min_change)
        ):
            entry.change = "regression"
            trend_percentage_list.append((float(trend_percentage), entry))

    return trend_percentage_list
