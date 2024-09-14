import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from pathlib import Path
from datetime import datetime
import calendar
import os
from enum import Enum
from typing import Tuple

from gpt import get_completion
from texts import txt
from helper import MONTHS_DICT, SEASON_DICT


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


class ReportType(Enum):
    DAILY = "day"
    MONTHLY = "month"
    YEARLY = "year"
    SEASON = "season"


RECORD_TYPE_DICT = {
    "poi": "POI",    
    "cp": "CP",
}

class ComparisonPeriod(Enum):
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    LAST_YEAR_MONTH = "last_year_month"
    LAST_5_YEARS_MONTH = "last_5_years_month"
    LAST_10_YEARS_MONTH = "last_10_years_month"
    ALL_MONTH = "all_time_month"

    THIS_SEASON = "this_season"
    LAST_SEASON = "last_season"
    LAST_5_YEARS_SEASON = "last_5_years_season"
    LAST_10_YEARS_SEASON = "last_10_years_season"
    ALL_SEASON = "all_time_season"

    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    LAST_5_YEARS = "last_5_years"
    LAST_10_YEARS = "last_10_years"
    ALL = "all_time"


class SEASON(Enum):
    SPRING = 1
    SUMMER = 2
    FALL = 3
    WINTER = 1


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
report_fields_old = [
    "record_type",
    "period",
    "period_value",
    "parameter",
    "value",
    "function",
    "event_date",
    "diff",
    "diff_perc",
    "significance",
]

report_fields = {
    'record_type': pd.Series(dtype='str'),       
    'period': pd.Series(dtype='str'),       
    'period_value': pd.Series(dtype='float'), 
    "parameter": pd.Series(dtype='str'),   
    "value": pd.Series(dtype='float'), 
    "function": pd.Series(dtype='str'), 
    "event_date": pd.Series(dtype='str'), 
    "diff": pd.Series(dtype='float'), 
    "diff_perc": pd.Series(dtype='float'), 
    "significance": pd.Series(dtype='str'), 
}



class Report:
    def __init__(self, parent, period, settings: dict) -> None:
        self.parent = parent
        self.period = period
        self.title = settings["title"]
        self.auto_min_days = settings["auto-min-days"]
        self.compare_to_periods = settings["compare-to-periods"]
        self.parameters = settings["parameters"]

    def get_significance(self, diff: float, std: float, stat_func: str) -> int:
        if stat_func == "mean":
            return 2 if abs(diff) > 2 * std else 1 if abs(diff) > std else 0
        elif stat_func == "min":
            return 2 if diff < 0 else 0
        elif stat_func == "max":
            return 2 if diff > 0 else 0

    def get_last_month_year(self, month, year):
        last_month = month - 1 if month > 1 else 12
        last_year = year if month > 1 else year - 1
        return last_month, last_year

    def get_last_season_year(self, season, year):
        last_season = season - 1 if season > 1 else 4
        last_year = year if season > 1 else year - 1
        return last_season, last_year

    def get_minmax_date(self, df, par, stat_func, stat_value):
        if stat_func in ["min", "max"]:
            filtered_df = df[df[par] == stat_value]
            if not filtered_df.empty:
                return filtered_df[self.parent.timestamp_col].iloc[0].strftime("%Y-%m-%d")
            else:
                return None
        else:
            return None
    def get_rank(self, value, period_df, par):
        pass

    def compare(self, par: str, df_report: pd.DataFrame, target_value: float):
        stat_func = get_stat_func(par)
        stats = [stat_func, "std"]
        # todo: review this carefully: there should be functions that take care of year(last_month), year(last_season) etc
        for period in self.compare_to_periods:
            if period == ComparisonPeriod.THIS_MONTH.value:
                month = self.parent.month
                years = [self.parent.year]
                period_value = f'{calendar.month_name[month]}/{self.parent.year}'
            elif period == ComparisonPeriod.LAST_MONTH.value:
                month, year = self.get_last_month_year(
                    self.parent.month, self.parent.year
                )
                years = [year]
                period_value = f'{calendar.month_name[month]}/{year}'
            elif period == ComparisonPeriod.THIS_SEASON.value:
                season = self.parent.season
                years = [self.parent.year]
                period_value = f'{SEASON_DICT[season]}/{self.parent.year}'
            elif period == ComparisonPeriod.LAST_SEASON.value:
                season, year = self.get_last_season_year(
                    self.parent.season, self.parent.year
                )
                years = [year]
                period_value = f'{SEASON_DICT[season]}/{year}'
            elif period == ComparisonPeriod.THIS_YEAR.value:
                years = [self.parent.year]
                period_value = f'{self.parent.year}'
            elif period == ComparisonPeriod.LAST_YEAR.value:
                years = [self.parent.year - 1]
                period_value = f'{self.parent.year - 1}'
            elif period == ComparisonPeriod.LAST_5_YEARS.value:
                years = range(self.parent.year - 6, self.parent.year)
                period_value = f'{self.parent.year - 6} - {self.parent.year - 1}'
            elif period == ComparisonPeriod.LAST_10_YEARS.value:
                years = range(self.parent.year - 11, self.parent.year)
                period_value = f'{self.parent.year - 11} - {self.parent.year - 1}'
            elif period == ComparisonPeriod.LAST_5_YEARS_MONTH.value:
                month = self.parent.month
                years = range(self.parent.year - 6, self.parent.year)
                period_value = f'{MONTHS_DICT[month]}/{self.parent.year - 6} - {MONTHS_DICT[month]}/{self.parent.year - 1}'
            elif period == ComparisonPeriod.LAST_10_YEARS_MONTH.value:
                month = self.parent.month
                years = range(self.parent.year - 11, self.parent.year)
                period_value = f'{MONTHS_DICT[month]}/{self.parent.year - 11} - {MONTHS_DICT[month]}/{self.parent.year - 1}'
            elif period == ComparisonPeriod.LAST_5_YEARS_SEASON.value:
                season = self.parent.season
                years = range(self.parent.year - 6, self.parent.year)
                period_value = f'{SEASON_DICT[season]}/{self.parent.year - 6} - {SEASON_DICT[season]}/{self.parent.year - 1}'
            elif period == ComparisonPeriod.LAST_10_YEARS_SEASON.value:
                season = self.parent.season
                years = range(self.parent.year - 11, self.parent.year)
                period_value = f'{SEASON_DICT[season]}/{self.parent.year - 11} - {SEASON_DICT[season]}/{self.parent.year - 1}'
            elif period == ComparisonPeriod.ALL_MONTH.value:
                month = self.parent.month
                years = range(self.parent.min_year, self.parent.max_year)
                period_value = f'{MONTHS_DICT[month]}/{self.parent.min_year} - {MONTHS_DICT[month]}/{self.parent.max_year - 1}'
            elif period == ComparisonPeriod.ALL_SEASON.value:
                season = self.parent.season
                years = range(self.parent.min_year, self.parent.max_year + 1)
                period_value = f'{SEASON_DICT[self.parent.season]} {self.parent.min_year} - {self.parent.max_year}'
            elif period == ComparisonPeriod.ALL.value:
                years = range(self.parent.min_year, self.parent.max_year + 1)
                period_value = f'{self.parent.min_year} - {self.parent.max_year}'
            
            if period in [
                ComparisonPeriod.THIS_MONTH.value,
                ComparisonPeriod.LAST_MONTH.value,
                ComparisonPeriod.LAST_5_YEARS_MONTH.value,
                ComparisonPeriod.LAST_10_YEARS_MONTH.value,
                ComparisonPeriod.ALL_MONTH.value,
            ]:
                period_df = self.parent.get_month_year_df(month=month, years=years)
                period_df_agg = (
                    period_df.groupby("month").agg({par: stats}).reset_index()
                )
            elif period in [
                ComparisonPeriod.THIS_SEASON.value,
                ComparisonPeriod.LAST_SEASON.value,
                ComparisonPeriod.LAST_5_YEARS_SEASON.value,
                ComparisonPeriod.LAST_10_YEARS_SEASON.value,
                ComparisonPeriod.ALL_SEASON.value,
            ]:
                period_df = self.parent.get_season_df(season=season, years=years)
                period_df_agg = (
                    period_df.groupby("season").agg({par: stats}).reset_index()
                )
            elif period in [
                ComparisonPeriod.THIS_YEAR.value,
                ComparisonPeriod.LAST_YEAR.value,
                ComparisonPeriod.LAST_5_YEARS.value,
                ComparisonPeriod.LAST_10_YEARS.value,
            ]:
                period_df = self.parent.get_years_df(years=years)
                period_df_agg = (
                    period_df.groupby("year").agg({par: stats}).reset_index()
                )
            elif period == ComparisonPeriod.ALL.value:
                period_df = self.parent.values_df
                period_df['dummy'] = 1
                period_df_agg = period_df.groupby("dummy").agg({par: stats}).reset_index()

            rows = []
            std = period_df_agg[(par, "std")].iloc[0]
            stat_value = period_df_agg[(par, stat_func)].iloc[0]
            diff = target_value - stat_value
            diff_perc = diff / stat_value * 100
            significance = self.get_significance(diff, std, stat_func)
            stat_date = self.get_minmax_date(period_df, par, stat_func, stat_value)
            # rank = self.get_rank(period_df, par, stat_func, stat_value)
            rows.append(
                pd.DataFrame(
                    columns=report_fields,
                    data=[
                        [
                            RECORD_TYPE_DICT['cp'],
                            period,
                            period_value,
                            par,
                            stat_value,
                            stat_func,
                            stat_date,
                            diff,
                            diff_perc,
                            significance,
                        ]
                    ],
                )
            )
            df_report = pd.concat([df_report] + rows, axis=0)
        return df_report

    def get_target_value(self, par)-> Tuple[float, pd.DataFrame]:
        
        if self.period == "day":
            period_data_df = self.parent.get_day_df(self.parent.date)
            value = period_data_df[par].iloc[0]
        elif self.period == "month":
            period_data_df = self.parent.get_month_year_df(self.parent.month, [self.parent.year])
            period_data_df['dummy'] = 1
            agg_df = period_data_df.groupby('dummy').agg(get_stat_func(par))
            value = agg_df[par].iloc[0]
        elif self.period == "season":
            period_data_df = self.parent.get_season_df(self.parent.season, [self.parent.year])
            period_data_df['dummy'] = 1
            
            agg_df = period_data_df.groupby('dummy').agg(get_stat_func(par))
            value = agg_df[par].iloc[0]
        elif self.period == "year":
            period_data_df = self.parent.get_years_df([self.parent.year])
            period_data_df['dummy'] = 1
            agg_df = period_data_df.groupby('dummy').agg(get_stat_func(par))
            value = agg_df[par].iloc[0]
        return value, period_data_df

    def get_title(self):
        """
        Returns the formatted title based on the specified period.

        Returns:
            str: The formatted title based on the specified period.
        """
        formats = {
            "day": lambda: self.title.format(self.parent.date.strftime("%Y-%m-%d")),
            "month": lambda: self.title.format(
                calendar.month_name[self.parent.month], self.parent.year
            ),
            "season": lambda: self.title.format(
                SEASON_DICT[self.parent.season], self.parent.year
            ),
            "year": lambda: self.title.format(self.parent.year),
        }
        return formats.get(self.period, lambda: None)()

    def get_report(self):
        report_df = pd.DataFrame(columns=report_fields)
        # daly, monthly, seasonal or yearly report
        period_value = {
            "day": self.parent.date,
            "month": self.parent.month,
            "season": self.parent.season,
            "year": self.parent.year,
        }[self.period]
        for par in self.parameters:
            # gets the daily value for a daily value, the mean month average for a monthly value etc
            target_value, period_df = self.get_target_value(par)
            stat_func = get_stat_func(par)
            stat_date = self.get_minmax_date(period_df, par, stat_func, target_value)
            row = pd.DataFrame(
                data=[
                    [
                        RECORD_TYPE_DICT["poi"],
                        self.period,
                        period_value,
                        par,
                        target_value,
                        stat_func,
                        stat_date,
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
        self.date = self.values_df[self.timestamp_col].dt.date.max()
        self.year = self.date.year
        self.month = self.date.month
        self.season = MONTH2SEASON[self.month]
        for key, report in settings["reports"].items():
            self.reports[key] = Report(self, key, report)
        self.ods_metadata = self.get_ods_metadata()

    @property
    def min_year(self):
        return self.values_df["year"].min()

    @property
    def max_year(self):
        return self.values_df["year"].max()

    @property
    def select_expression(self):
        return f"select={self.timestamp_col}, {','.join(self.value_cols)}"

    def get_odsmon_column_names(self):
        """
        Returns the column names of the value DataFrame.

        Returns:
            list: A list of column names.
        """
        return self.value_df.columns
    
    def get_report_types(self):
        return list(self.reports.keys())

    def get_year_range(self):
        return range(self.min_year, self.max_year + 1)

    def get_date_range(self):
        return (self.values_df[self.timestamp_col].min(), self.values_df[self.timestamp_col].max())
    
    def get_num_records(self):
        return len(self.values_df)
    
    def complete_data(self):
        self.values_df[self.timestamp_col] = pd.to_datetime(
            self.values_df[self.timestamp_col]
        )
        self.values_df["year"] = self.values_df[self.timestamp_col].dt.year
        self.values_df["month"] = self.values_df[self.timestamp_col].dt.month
        self.values_df["season"] = self.values_df["month"].map(MONTH2SEASON)
        self.values_df["season_year"] = np.where(
            self.values_df["month"].isin((1, 2)),
            self.values_df["year"] + 1,
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
        """
        Generates the URL for retrieving all data from the dataset.

        Parameters:
            where (str): Optional filter condition for the data.

        Returns:
            str: The URL for retrieving all data from the dataset.
        """
        return f"https://{self.root}/api/explore/v2.1/catalog/datasets/{self.key}/exports/csv?lang=de&timezone=UTC&use_labels=false&delimiter=%3B&{self.select_expression}&where={where}"

    def open_data_file(self):
        """
        Opens the data file and loads it into a pandas DataFrame.

        Returns:
            bool: True if the data file is successfully loaded, False otherwise.
        """
        try:
            self.values_df = pd.read_parquet(self.data_file)
            return True
        except Exception as e:
            st.warning(f"Failed to load data file: {e}")
            return False

    def format_raw_datat(self, df_raw):
        """
        Formats the raw data by performing the following steps:
        1. Converts the timestamp column to datetime format.
        2. Extracts the date from the timestamp column.
        3. Groups the data by the timestamp column and applies aggregation functions specified in `value_cols`.
        4. Renames the columns of the grouped dataframe by joining the column names with '_'.
        5. Corrects the aggregation column name which is not multi-indexed.
        6. Sorts the grouped dataframe by the timestamp column in descending order.
        7. Converts the timestamp column back to datetime format.

        Parameters:
            df_raw (pandas.DataFrame): The raw data to be formatted.

        Returns:
            pandas.DataFrame: The formatted data.
        """
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
        """
        Initializes the data file by loading the data from a pqrquet file. if no file is found
        the data is loaded from the specified ODS-URL for the dataset, formatted and then saved
        to the parquet file.

        Returns:
            bool: True if the data is successfully loaded, False otherwise.
        """
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
        """
        Synchronizes the data file with the latest data from the specified URL.
        Returns:
            None
        Raises:
            None
        """
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
        """
        Save the values dataframe as a parquet file.

        Parameters:
        - None

        Returns:
        - None
        """

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

    def get_reports(
        self, type: ReportType, generate_summary: bool
    ) -> Tuple[str, pd.DataFrame, str]:
        """
        Retrieves reports of the specified type (daily, monthly, yearly...).
        Args:
            type (ReportType): The type of report to retrieve.
        Returns:
            Tuple[str, pd.DataFrame, str]: A tuple containing the title of the report, the report data as a pandas DataFrame, and a text string.
        """
        df = self.reports[type].get_report()
        summary_text = (
            get_completion(
                txt["system_prompt"],
                txt["user_prompt"].format(df.to_string(index=False)),
            )
            if generate_summary
            else ""
        )

        return self.reports[type].get_title(), df, summary_text

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
