import pandas as pd
from config import DB_CONNECTION_STRING
from datetime import timedelta


def load_data(count="1000", event_type="view_item"):
    query = f"SELECT * FROM events WHERE event_type = '{event_type}' ORDER BY timestamp DESC LIMIT {count}"
    return pd.read_sql(query, DB_CONNECTION_STRING)
    try:
        return pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None


def add_sessions(df):
    df = df.sort_values(by=["customer_id", "timestamp"])

    df["session_id"] = (
        (df["customer_id"] != df["customer_id"].shift())
        | (df["timestamp"] - df["timestamp"].shift() > pd.Timedelta(hours=1))
    ).cumsum()

    return df


def get_popular_items(df, count=10):
    most_popular = (
        df.groupby(["customer_id", "product_id"])
        .agg({"customer_id": "first", "product_id": "first"})["product_id"]
        .value_counts()
        .head(count)
        .keys()
        .tolist()
    )

    return most_popular


df = load_data(1000)
df = add_sessions(df)
