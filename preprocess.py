import pandas as pd
import uuid
from config import DB_CONNECTION_STRING
from datetime import timedelta

from utils import get_timestamp


def add_session_ids(df):
    df = df.sort_values(by=["customer_id", "timestamp"])

    df["session_id"] = (
        (df["customer_id"] != df["customer_id"].shift())
        | (df["timestamp"] - df["timestamp"].shift() > pd.Timedelta(hours=1))
    ).cumsum()

    df["session_id"] = df.groupby("session_id")["session_id"].transform(
        lambda x: uuid.uuid4()
    )

    return df


def remove_short_sessions(df, threshold=2):
    session_lengths = df.groupby("session_id").size()
    sessions_to_keep = session_lengths[session_lengths >= threshold].index
    return df[df["session_id"].isin(sessions_to_keep)]


def separate_last_user_events(df):
    last_user_id = df.iloc[-1]["customer_id"]
    last_user_events = df[df["customer_id"] != last_user_id]

    return df[df["customer_id"] != last_user_id], df[df["customer_id"] == last_user_id]


def preprocess_events():
    table_name = f"preprocessed_events_{get_timestamp()}"
    last_user_events = None

    query = "SELECT * FROM events WHERE event_type = 'view_item' ORDER BY timestamp"
    for chunk in pd.read_sql(query, DB_CONNECTION_STRING, chunksize=100000):
        df = pd.concat([last_user_events, chunk], ignore_index=True)
        df, last_user_events = separate_last_user_events(df)
        df = add_session_ids(df)
        df = remove_short_sessions(df)
        df.to_sql(table_name, DB_CONNECTION_STRING, if_exists="append", index=False)

    return table_name


def get_product_counts(df):
    product_counts_df = (
        df.groupby(["customer_id", "product_id"])
        .agg({"customer_id": "first", "product_id": "first"})["product_id"]
        .value_counts()
    )
    return product_counts_df


def preprocess_products():
    table_name = f"product_counts_{get_timestamp()}"

    s = pd.Series(dtype=int)
    query = f"SELECT * FROM events"
    for chunk in pd.read_sql(query, DB_CONNECTION_STRING, chunksize=100000):
        s = s.add(get_product_counts(chunk), fill_value=0).astype(int)

    s.rename("count").to_sql(table_name, DB_CONNECTION_STRING, index_label="product_id")

    return table_name


def preprocess():
    preprocess_events()
    preprocess_products()


preprocess()