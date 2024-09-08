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
    "user_prompt": """Please provide a summary of the reported values in 3 to 5 sentences. : 
    {}
    """,
    "info": """# 📈ODS-MON
OGD Data provide increasingly high resolution time series data. Wthin the ODS environment, such data is difficult to analyse. While data is updated daily or even hourly, most user do not wish to inspect this data continously and only wish to be averted of significant changes. Such changes are generally exeptionally high or low values as compared to historic data or exceedances of regulatory standards.

ODS-Mon allows to configure reports for timeseries, that allow to extract essential information and uses a Large Language Model to summarize the findings in natural language. Reports include daily reports, comparing a selected date to various periods, for example the last month, or year to date, or the last 1, 5 or 10 years. For each period to compare to, mean, min and max values are compared to the mean, min and max value for the day in question. The monthly report selects the month before the selected date and compares its statistics with the defines comparison periods as in the previous daily report. Yearly reports compare yearly statistics with previous periods, typically the last previous 1, 5 or 10 years or all time.

""",
}
