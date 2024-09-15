# 📈 Data-News-Generator (DNG)

## Introduction
The **Data-News-Generator (DNG)** app allows the aggregation of time series data hosted on the [Opendatasoft](https://www.opendatasoft.com/) platform across various periods (e.g., month, season, year) and enables comparison with historical data. You can try the app [here](https://ods-mon.streamlit.app/) or install it locally on your machine. The current version does not allow for the configuration of datasets directly through the app, so if you wish to analyze specific ODS datasets, you will need to install the app as described below.

## Installation

To get started with the Data-News-Generator app, follow these steps to install it locally on your machine:

### Prerequisites
- Python 3.8+
- pip

### Steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/lcalmbach/ods-mon.git
   cd ods-mon
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Feature Overview

**Data-News-Generator (DNG)** automates the generation of insightful reports based on high-frequency time series data, particularly from **Open Government Data (OGD)** sources like [Opendatasoft](https://www.opendatasoft.com/) (ODS). DNG helps identify significant data changes, such as unusually high or low values, by comparing them to historical records or regulatory standards.

### Key Features:
- **Automated report generation** for periods such as days, months, years, and seasons.
- **Comparative analysis** based on historical datasets (e.g., last year, last 10 years).
- **Statistical insights** including mean, max, and min values for selected parameters.
- **AI-powered summaries** generated using the GPT-4 language model, providing natural language insights based on report data.

### Example Use Case:
Imagine you're tracking air quality data, focusing on ozone levels. You want to compare the mean, minimum, and maximum values of ozone for the current day, month, or year with historical data, such as the last month's averages, the values from the same month over the past 5 years, or any extreme values recorded in past years. Instead of manually downloading datasets and generating statistics, you can configure the **monitor.json** file to automatically generate daily, seasonal, monthly, or yearly reports. After editing the file, the reports become available in the app’s GUI, and both a report table and an AI-generated summary are produced by clicking the `Generate Report` button.

---

## Configuration File

The DNG app relies on a configuration file to define how reports should be generated. Below is a detailed explanation of each section of the settings file:

```json
{
    "100051": {
        "root": "data.bs.ch",
        "dataset": "100051",
        "title": "Luftqualität Station Basel-Binningen",
        "timestamp-col": "datum_zeit",
        "include-is-weekday": false,
        "value-cols": {
            "o3_ug_m3": {
                "name": "O3",
                "agg_funcs": ["mean", "max", "min"],
                "standards": []
            },
            "no2_ug_m3": {
                "name": "NO2",
                "agg_funcs": ["mean", "max", "min"],
                "standards": []
            },
            "pm10_ug_m3": {
                "name": "PM10",
                "agg_funcs": ["mean", "max", "min"],
                "standards": []
            },
            "pm2_5_ug_m3": {
                "name": "PM2.5",
                "agg_funcs": ["mean", "max", "min"],
                "standards": []
            }
        },
        "aggregation": "d",
        "rename-cols": false,
        "reports": {
            "day": {
                "title": "Vergleich verschiedener Luftqualitätsparameter vom {} mit historischen Werten",
                "auto-min-days": 1,
                "compare-to-periods": ["last_month", "year_todate", "all_time"],
                "parameters": ["o3_ug_m3_mean", "o3_ug_m3_max", "no2_ug_m3_mean", "no2_ug_m3_max", "pm10_ug_m3_mean", "pm10_ug_m3_max", "pm2_5_ug_m3_mean", "pm2_5_ug_m3_max"]
            },
            "month": {
                "title": "Vergleich verschiedener monatlicher Luftqualitätsparameter vom {} {} mit historischen Werten",
                "auto-min-days": 15,
                "compare-to-periods": ["last_year", "month_all_time"],
                "parameters": ["o3_ug_m3_mean", "o3_ug_m3_max", "no2_ug_m3_mean", "no2_ug_m3_max", "pm10_ug_m3_mean", "pm10_ug_m3_max", "pm2_5_ug_m3_mean", "pm2_5_ug_m3_max"]
            },
            "year": {
                "title": "Vergleich der jährlichen Luftqualitätsparameter von {} mit historischen Werten",
                "auto-min-days": 90,
                "compare-to-periods": ["last_year", "all_time"],
                "parameters": ["o3_ug_m3_mean", "o3_ug_m3_max", "no2_ug_m3_mean", "no2_ug_m3_max", "pm10_ug_m3_mean", "pm10_ug_m3_max", "pm2_5_ug_m3_mean", "pm2_5_ug_m3_max"]
            },
            "season": {
                "title": "Vergleich der saisonalen Luftqualitätsparameter von {} mit historischen Werten",
                "auto-min-days": 30,
                "compare-to-periods": ["last_year", "all_time"],
                "parameters": ["o3_ug_m3_mean", "o3_ug_m3_max", "no2_ug_m3_mean", "no2_ug_m3_max", "pm10_ug_m3_mean", "pm10_ug_m3_max", "pm2_5_ug_m3_mean", "pm2_5_ug_m3_max"]
            }
        }
    }
}
```

### Configuration Details:

- **`root`**: The root domain of the ODS data source (e.g., `data.bs.ch` for Basel's open data platform or `data.bl.ch` for Baselland's OGD platform).
- **`dataset`**: The specific ODS dataset ID, such as `100051`, representing the air quality data for Basel-Binningen.
- **`timestamp-col`**: The column in the dataset representing the timestamp (`datum_zeit`).
- **`include-is-weekday`**: A boolean indicating whether to include data based on weekdays. This feature is not yet implemented but is intended to generate separate statistics for workdays and weekends (useful for traffic or energy data).
- **`value-cols`**: A dictionary representing the data points of interest (e.g., `O3`, `NO2`, `PM10`). Each parameter includes:
  - **`name`**: A human-readable name for the parameter. This is useful if the dataset contains cryptic field names, as ODS labels are not imported directly.
  - **`agg_funcs`**: Aggregation functions applied when the dataset has higher frequency data (e.g., hourly measurements). The app aggregates values like `O3_mean`, `O3_max`, and `O3_min` from daily measurements.
  - **`standards`**: A placeholder for future regulatory standards or limits.
- **`aggregation`**: Defines the level of aggregation (e.g., daily (`d`), weekly, monthly).
- **`rename-cols`**: Boolean indicating whether to rename columns.
  
### Reports Configuration:
Each report type (daily, monthly, yearly, or seasonal) includes the following parameters:

- **`title`**: The title of the report, dynamically populated with relevant dates.
- **`auto-min-days`**: The minimum number of days required to generate a report.
- **`compare-to-periods`**: The periods to compare against. Here are the available options:
    - `this_month`: All values for the current month
    - `last_month`: The previous full month
    - `last_year_month`: The same month last year
    - `last_5_years_month`: The same month for the past 5 years (e.g., average August temperature over 5 years)
    - `last_10_years_month`: The same month for the past 10 years
    - `all_time_month`: The selected month over all available years in the dataset
    - `this_season`: The current season
    - `last_season`: The last complete season
    - `last_5_years_season`: The last complete season over the past 5 years
    - `last_10_years_season`: The last complete season over the past 10 years
    - `all_time`: compare to all data available

- **parameters**: The specific columns and aggregation functions to be included in the report (e.g., o3_ug_m3_mean, pm2_5_ug_m3_max).

---

## Future Plans

- **Email Subscriptions**: Users will be able to subscribe to specific reports and receive notifications by email daily, monthly and yearly. Daily reports can be limited to days, when significant changes are detected.
  
---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.