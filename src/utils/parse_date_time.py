import pandas as pd

def parse_date_time(df: pd.DataFrame, date_time_col: str, time_col: str,
                    date_col: str):
    df[date_time_col] = pd.to_datetime(df[date_time_col], format="mixed")
    df[time_col] = df[date_time_col].dt.hour
    df[date_col] = df[date_time_col].dt.date
    return df


