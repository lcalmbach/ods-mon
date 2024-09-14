txt = {
    "system_prompt": """Your are a data expert. you are given the following information:
- a list of parameters
- a period expression: day:
- a table with various comparison data-values.

the table has the following columns:
- date: date for report
- period: compare a value the given period, which can be: last_month, year to date, last year, last  years. last 10 years etc. 
- period_value: vor example if the raget period is day (2024-09-10) and the compare period is last_year then the period_value 2023
- parameter: parameter name. usually there are 3 columns per parameter: xxx_min, xxx_max, xxx_mean
- value: the value for the given parameter. the first row holds the value for the target period, the following rows hold values for the compare-periods, e.g. mean of last year. note that if the target value is mean, then mean is also used for the calculated statistics. 
- function: statistical function used to calculate the value. usually there are 3 columns per parameter: xxx_min, xxx_max, xxx_mean
- rank: not used yet
- diff: difference between the target value and the compare value
- diff_percent: difference in percent between the target value and the compare value
- significance: for a mean value, a difference is considered moderate if the difference is larger than the standard deviation of the compare values and significant if the difference is larger than 2x the standard deviation.


""",
    "table_legend": """POI: Period of interest</br>
    CP: Comparison Period""",
    "user_prompt": """Please provide a summary of the reported values in 3 to 5 sentences. : 
    {}
    """,
    "info": """# 📈 Data-News-Generator (DNG)
OGD (Open Government Data) providers are offering increasingly high-resolution time series data. While this data is updated daily or even hourly, most users are not interested in continuous monitoring and instead prefer to be notified only when significant changes occur. These changes typically involve exceptionally high or low values compared to historical data (e.g., last year, last 10 years, same month in the last 10 years) or regulatory standards. Manually generating reports to track such changes can be simple but tedious.

The Data-News-Generator (DNG) app provides a framework to automate the generation of such reports based on a configuration file. Currently, the only supported data provider is [Opendatasoft](https://www.opendatasoft.com/) (ODS), a popular platform for publishing datasets, particularly in the Open Government Data community. ODS offers a powerful API for accessing data and metadata.

A typical DNG report contains the following minimal configuration details:

- The dataset to be used
- The parameters to be included in the report
- The period of interest (e.g., last day, month, year, etc.)
- Comparison periods (e.g., last year, last 10 years, same month in the last 10 years, etc.)
- Aggregation functions for comparison (e.g., mean, min, max, etc.)
- The output presents the selected parameters for the period of interest alongside the comparison periods and calculates the differences. A significance factor is computed for each comparison, indicating whether the difference is notable. A factor of 1 is assigned if the difference between the period of - interest and the comparison period exceeds one standard deviation of the comparison period. A factor of 2 is assigned if the difference exceeds two standard deviations or if the value is higher than the maximum or lower than the minimum of the comparison period.

The report is generated in a tabular format, and a summary is created using a Large Language Model (LLM). The summary is written in natural language and highlights the key insights from the report. Currently, it utilizes the OpenAI API with the GPT-4 model (referred to as GPT-4o). This functionality is still under development, and while the generated summaries are generally accurate, the model sometimes struggles to differentiate between relevant and irrelevant information.

The app is designed to be extensible, future version should allow users to subsribe to selected reports. Reports are then received by mail and the user may define conditions for when a report should be mailed to them: for example daily reports can be sent daily or just if significant changes have been detected. It is also planned to included additional data providers.
""",

    "settings_info": """Below are the settings for the reports in JSON format that can be generated. Each node corresponds to an ODS dataset, and for each dataset, reports can be generated on a daily, monthly, or yearly basis.""",
}
