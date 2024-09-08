import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from pathlib import Path
from datetime import datetime
import os
from enum import Enum
from data_gpt import get_completion

from texts import txt


def get_stat_func(par: str) -> str:
    """
    Returns the statistical function based on the given parameter.

    Parameters:
    par (str): The parameter to determine the statistical function.

    Returns:
    str: The statistical function. Possible values are "min", "max", "sum", or "mean".
    """
    if par[-4:] == "_min":
        return "min"
    elif par[-4:] == "_max":
        return "max"
    elif par[-4:] == "_sum":
        return "sum"
    else:
        return "mean"


class ComparisonPeriod(Enum):
    YESTERDAY = "yesterday"
    LAST_MONTH = "last_month"
    LAST_SEASON = "last_season"
    YEAR_TODATE = "year_todate"
    LAST_YEAR = "last_year"
    LAST_5_YEARS = "last_5_years"
    LAST_10_YEARS = "last_10_years"
    ALL = "all_time"
    LAST_YEAR_MONTH = "last_year_month"
    LAST_5_YEARS_MONTH = "last_5_years_month"
    LAST_10_YEARS_MONTH = "last_10_years_month"
    ALL_MONTH = "all_time_month"


class SEASON(Enum):
    WINTER = 1
    SPRING = 2
    SUMMER = 3
    FALL = 4


SEASON_MONTHS = {1: [12, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9, 10, 11]}
MONTH2SEASON = {
    12: SEASON.WINTER.value,
    1: SEASON.WINTER.value,
    2: SEASON.WINTER.value,
    3: SEASON.SPRING.value,
    4: SEASON.SPRING.value,
    5: SEASON.SPRING.value,
    6: SEASON.SUMMER.value,
    7: SEASON.SUMMER.value,
    8: SEASON.SUMMER.value,
    9: SEASON.FALL.value,
    10: SEASON.FALL.value,
    11: SEASON.FALL.value,
}

# columns for the report table
report_fields = [
    "date",
    "period",
    "period_value",
    "parameter",
    "value",
    "function",
    "event_date",
    "rank",
    "diff",
    "diff_perc",
    "significance",
]


class Report:
    def __init__(self, parent, period, settings: dict) -> None:
        self.parent = parent
        self.period = period
        self.refresh_comparison_interval()
        self.compare_to_periods = settings["compare_to_periods"]
        self.parameters = settings["parameters"]

    def refresh_comparison_interval(self):
        self.date = self.parent.date
        self.this_month = self.date.month
        self.this_year = self.date.year
        self.this_season = MONTH2SEASON[self.this_month]
        self.last_month = self.this_month - 1 if self.this_month > 1 else 12
        self.last_year = self.this_year - 1
        self.last_season = self.this_season - 1 if self.this_season > 1 else 4

    def get_significance(self, diff: float, std: float, stat_func: str) -> int:
        if stat_func == "mean":
            return 2 if abs(diff) > 2 * std else 1 if abs(diff) > std else 0
        elif stat_func == "min":
            return 2 if diff < 0 else 0
        elif stat_func == "max":
            return 2 if diff > 0 else 0

    def compare(self, par: str, df_report: pd.DataFrame, target_value: float):
        stat_func = get_stat_func(par)
        stats = [stat_func, "std"]
        for period in self.compare_to_periods:
            if period == ComparisonPeriod.LAST_MONTH.value:
                period_value = self.this_month
                period_df = self.parent.get_month_year_df(
                    month=self.this_month, years=[self.this_year]
                )
                period_df_agg = (
                    period_df.groupby("month").agg({par: stats}).reset_index()
                )
            elif period == ComparisonPeriod.YEAR_TODATE.value:
                period_value = self.this_year
                period_df = self.parent.get_years_df(years=[self.this_year])
                period_df_agg = (
                    period_df.groupby("year").agg({par: stats}).reset_index()
                )
            elif period == ComparisonPeriod.LAST_YEAR.value:
                period_value = self.this_year
                period_df = self.parent.get_years_df(years=[self.this_year])
                period_df_agg = (
                    period_df.groupby("year").agg({par: stats}).reset_index()
                )
            elif period == ComparisonPeriod.ALL.value:
                period_value = f"{self.parent.min_year}-{self.parent.max_year}"
                period_df = self.parent.values_df
                period_df["x"] = 1
                period_df_agg = period_df.groupby("x").agg({par: stats}).reset_index()

            rows = []
            std = period_df_agg[(par, "std")].iloc[0]
            stat_value = period_df_agg[(par, stat_func)].iloc[0]
            diff = target_value - stat_value
            diff_perc = diff / stat_value * 100
            significance = self.get_significance(diff, std, stat_func)

            rows.append(
                pd.DataFrame(
                    columns=report_fields,
                    data=[
                        [
                            self.date,
                            period,
                            period_value,
                            par,
                            stat_value,
                            stat_func,
                            None,
                            0,
                            diff,
                            diff_perc,
                            significance,
                        ]
                    ],
                )
            )
            df_report = pd.concat([df_report] + rows, axis=0)
        return df_report

    def get_target_value(self, par):
        if self.period == "day":
            value_dict = self.parent.get_day_df(self.date).iloc[0].to_dict()
            return value_dict[par]
        elif self.period == "month":
            _df = self.parent.get_month_year_df(self.this_month, [self.this_year])
            return _df[par].agg(get_stat_func(par))
        elif self.period == "season":
            return (
                self.parent.get_season_df(self.this_season, [self.this_year])
                .iloc[0]
                .to_dict()
            )
        elif self.period == "year":
            value_dict = self.parent.get_years_df([self.this_year]).iloc[0].to_dict()
            return value_dict[par]

    def get_report(self):
        self.refresh_comparison_interval()
        report_df = pd.DataFrame(columns=report_fields)
        period_value = {
            "day": self.date,
            "month": self.this_month,
            "season": self.this_season,
            "year": self.this_year,
        }[self.period]
        for par in self.parameters:
            target_value = self.get_target_value(par)
            row = pd.DataFrame(
                data=[
                    [
                        self.date,
                        self.period,
                        period_value,
                        par,
                        target_value,
                        get_stat_func(par),
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                ],
                columns=report_fields,
            )
            report_df = pd.concat([report_df, row])
            report_df = self.compare(par, report_df, target_value)
        return report_df


class Dataset:
    def __init__(self, key, settings) -> None:
        self.root = settings["root"]
        self.title = settings["title"]
        self.key = key
        self.timestamp_col = settings["timestamp-col"]
        self.value_cols = settings["value-cols"]
        self.values_df = pd.DataFrame()
        self.data_root = Path("data")
        self.aggregation = settings["aggregation"]
        self.include_is_weekday = settings["include-is-weekday"]
        self.reports = {}
        self.rename_columns = settings["rename-cols"]
        self.is_synched = False
        self.data_file = self.data_root / f"{self.key}.parquet"
        if os.path.exists(self.data_file):
            if self.open_data_file():
                self.values_df = pd.read_parquet(self.data_file)
                self.is_synched = self.synch_data_file()
            else:
                st.warning(f"Failed to load data file {self.data_file}")
        else:
            if self.init_data_file():
                st.success(f"Data file {self.data_file} created successfully")
        self.save()
        self.complete_data()
        self._date = self.values_df[self.timestamp_col].dt.date.max()
        for key, report in settings["reports"].items():
            self.reports[key] = Report(self, key, report)
        self.ods_metadata = self.get_ods_metadata()

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        self._date = value
        for key, report in self.reports.items():
            self.reports[key].date = self.date

    @property
    def min_year(self):
        return self.values_df["year"].min()

    @property
    def max_year(self):
        return self.values_df["year"].max()

    @property
    def select(self):
        return f'select={self.timestamp_col}, {",".join(self.value_cols)}'

    def complete_data(self):
        self.values_df[self.timestamp_col] = pd.to_datetime(
            self.values_df[self.timestamp_col]
        )
        self.values_df["year"] = self.values_df[self.timestamp_col].dt.year
        self.values_df["month"] = self.values_df[self.timestamp_col].dt.month
        self.values_df["season"] = self.values_df["month"].map(MONTH2SEASON)
        self.values_df["season_year"] = np.where(
            self.values_df["month"] == 12,
            self.values_df["year"] - 1,
            self.values_df["year"],
        )
        if self.include_is_weekday:
            self.values_df["is_weekday"] = (
                self.values_df[self.timestamp_col].dt.dayofweek < 5
            )

        if self.rename_columns:
            rename_dict = {key: value["name"] for key, value in self.value_cols.items()}
            # Assuming you have a DataFrame 'df' with columns that match the keys in the dict
            self.values_df = self.values_df.rename(columns=rename_dict)
            # the column names need to be changed in the value_cols dict as well
            temp_cols = {data["name"]: data for key, data in self.value_cols.items()}
            self.value_cols = temp_cols

        self.values_df = self.values_df.sort_values(
            by=self.timestamp_col, ascending=False
        )
        return True

    def url_all_data(self, where: str = ""):
        return f"https://{self.root}/api/explore/v2.1/catalog/datasets/{self.key}/exports/csv?lang=de&timezone=UTC&use_labels=false&delimiter=%3B&{self.select}&where={where}"

    def open_data_file(self):
        try:
            self.values_df = pd.read_parquet(self.data_file)
            return True
        except Exception as e:
            st.warning(f"Failed to load data file: {e}")
            return False

    def format_raw_datat(self, df_raw):
        if len(df_raw) > 0:
            df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col])
            df_raw[self.timestamp_col] = df_raw[self.timestamp_col].dt.date
            agg_dict = {}
            for key, par in self.value_cols.items():
                if len(par["agg_funcs"]) > 0:
                    agg_dict[key] = par["agg_funcs"]
                    df_grouped = (
                        df_raw.groupby(self.timestamp_col).agg(agg_dict).reset_index()
                    )
                    df_grouped.columns = ["_".join(col) for col in df_grouped.columns]
                    # correct the aggregation column which is not multi indexed
                    df_grouped.rename(
                        columns={f"{self.timestamp_col}_": self.timestamp_col},
                        inplace=True,
                    )
                    df_grouped.sort_values(
                        self.timestamp_col, ascending=False, inplace=True
                    )
                    df_grouped[self.timestamp_col] = pd.to_datetime(
                        df_grouped[self.timestamp_col]
                    )
                else:
                    df_grouped = df_raw
            return df_grouped
        else:
            return pd.DataFrame()

    def init_data_file(self):
        with st.spinner(f"Loading {self.title} data..."):
            today_str = pd.Timestamp.today().strftime("%Y/%m/%d")
            where = f"{self.timestamp_col}<'{today_str}'"
            url = self.url_all_data(where)
            response = requests.get(url)
            if response.status_code == 200:
                csv_content = io.StringIO(response.text)
                df_raw = pd.read_csv(csv_content, delimiter=";")
                self.values_df = self.format_raw_datat(df_raw)
                return True
            else:
                st.warning(
                    f"Failed to retrieve content. Status code: {response.status_code}"
                )
                return False

    def synch_data_file(self):
        with st.spinner(f"Synching {self.title} data..."):
            max_timestamp = self.values_df[self.timestamp_col].max() + pd.Timedelta(
                days=1
            )
            max_timestamp_str = max_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            today_str = pd.Timestamp.today().strftime("%Y/%m/%d")
            where = f"{self.timestamp_col}<'{today_str}' AND {self.timestamp_col}>'{max_timestamp_str}'"
            url = self.url_all_data(where)
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    csv_content = io.StringIO(response.text)
                    df = pd.read_csv(csv_content, delimiter=";")
                    df = self.format_raw_datat(df)
                    if len(df) > 0:
                        self.values_df = pd.concat([self.values_df, df])
                else:
                    st.warning(
                        f"Failed to synchronize files. Status code: {response.status_code}"
                    )
            except Exception as e:
                st.warning(f"Failed to synchronize files or no new data available: {e}")

    def save(self):
        self.values_df.to_parquet(
            self.data_root / f"{self.key}.parquet", engine="pyarrow"
        )
        # uncomment to save as csv for debugging
        # self.values_df.to_csv(self.data_root / f'{self.key}.csv',index=False, sep=';')

    def get_last_date(self):
        return self.values_df[self.timestamp_col].max()

    def get_last_n_records(self, num_days):
        # previous attempt, reading the last record from the API
        # url = f'https://{self.root}/api/explore/v2.1/catalog/datasets/{self.key}/records?order_by={self.timestamp_col}%20desc&limit=1&timezone=Europe%2FBerlin'
        date = datetime.combine(self.date, datetime.min.time())
        return self.values_df[self.values_df[self.timestamp_col] <= date].head(num_days)

    def get_day_df(self, date):
        return self.values_df[self.values_df[self.timestamp_col].dt.date == date]

    def get_month_year_df(self, month, years):
        return self.values_df[
            (self.values_df["month"] == month) & (self.values_df["year"].isin(years))
        ]

    def get_years_df(self, years):
        return self.values_df[(self.values_df["year"].isin(years))]

    def get_season_df(self, season, years):
        return self.values_df[
            (self.values_df["season_year"].isin(years))
            & (self.values_df["season"] == season)
        ]

    def get_stats(self, df, col):
        return {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
        }

    def get_reports(self):
        reports = []
        for key, report in self.reports.items():
            df = report.get_report()
            text = get_completion(
                txt["system_prompt"],
                txt["user_prompt"].format(df.to_string(index=False)),
            )
            reports.append((key, df, text))

        return reports

    def home_url(self):
        return f"https://{self.root}/explore/dataset/{self.key}/"

    def get_ods_metadata(self):
        url = f"https://{self.root}/api/explore/v2.1/catalog/datasets/{self.key}?timezone=UTC&include_links=false&include_app_metas=false"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {}

    def description():
        pass

    def __str__(self) -> str:
        return self.title
