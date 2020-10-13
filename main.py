import pandas as pd
from config import DB_CONNECTION_STRING
from datetime import timedelta


def load_data(table_name="preprocessed_events", count="1000"):
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {count}"
    try:
        return pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None


def get_popular_items(table_name="product_counts", count=10):
    query = f"SELECT product_id FROM {table_name} ORDER BY count DESC LIMIT {count}"
    try:
        df = pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None

    return df["product_id"].tolist()


print(get_popular_items("product_counts_1602591623"))