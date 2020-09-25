import pandas as pd
from config import DB_CONNECTION_STRING


def load_data(count="10000", event_type="view_item"):
    query = f"SELECT * FROM events WHERE event_type = '{event_type}' ORDER BY timestamp DESC LIMIT {count}"
    return pd.read_sql(query, DB_CONNECTION_STRING)
    try:
        return pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None


df = load_data()

most_popular = (
    df.groupby(["customer_id", "product_id"])
    .agg({"customer_id": "first", "product_id": "first"})["product_id"]
    .value_counts()
    .head(10)
    .keys()
    .tolist()
)
