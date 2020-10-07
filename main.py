import pandas as pd
from config import DB_CONNECTION_STRING
from datetime import timedelta


def load_data(count="1000", table_name="preprocessed_events"):
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {count}"
    try:
        return pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None


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
