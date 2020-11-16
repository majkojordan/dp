import pandas as pd
import uuid
from config import DB_CONNECTION_STRING
from datetime import timedelta
from tqdm import tqdm

from utils import get_timestamp


def sql_to_csv(query, path):
    # sql_to_csv("SELECT * FROM preprocessed_events_month_1603127051 limit 1000", 'data/events_month_1000.csv')
    df = pd.read_sql(
        query,
        DB_CONNECTION_STRING,
    )
    df.to_csv(path)


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


def remove_unfrequent_items(df, threshold=10):
    # product_counts = df.groupby("product_id").size()
    # products_to_keep = product_counts[product_counts >= threshold].index

    # category_counts = df.groupby("categories").size()
    # categories_to_keep = category_counts[category_counts >= threshold].index
    # return df[
    #     df["product_id"].isin(products_to_keep)
    #     & df["categories"].isin(categories_to_keep)
    # ]
    product_counts = df.groupby("product_id").size()
    products_to_keep = product_counts[product_counts >= threshold].index
    return df[df["product_id"].isin(products_to_keep)]


def separate_last_user_events(df):
    last_user_id = df.iloc[-1]["customer_id"]
    last_user_events = df[df["customer_id"] != last_user_id]

    return df[df["customer_id"] != last_user_id], df[df["customer_id"] == last_user_id]


def transform_to_sessions(df):
    df = df.sort_values(by=["customer_id", "timestamp"])

    df["session_id"] = (
        (df["customer_id"] != df["customer_id"].shift())
        | (df["timestamp"] - df["timestamp"].shift() > pd.Timedelta(hours=1))
    ).cumsum()

    s = df.groupby("session_id")["product_id"].apply(list)

    # delete short sessions
    s = s[s.apply(lambda x: len(x) > 1)]

    return s.rename("click_sequence")


def preprocess_events():
    timestamp = get_timestamp()
    last_user_events = None

    query = "SELECT * FROM events WHERE event_type = 'view_item' ORDER BY customer_id"
    for chunk in tqdm(pd.read_sql(query, DB_CONNECTION_STRING, chunksize=100000)):
        df = pd.concat([last_user_events, chunk], ignore_index=True)
        df, last_user_events = separate_last_user_events(df)
        df = remove_unfrequent_items(df, 10)
        # df = add_session_ids(df)
        # df = remove_short_sessions(df)
        # df.to_sql(f"preprocessed_events_{timestamp}", DB_CONNECTION_STRING, if_exists="append", index=False)
        s = transform_to_sessions(df)
        s.to_sql(
            f"click_sequences_{timestamp}",
            DB_CONNECTION_STRING,
            if_exists="append",
            index=False,
        )


def preprocess_events_sample():
    timestamp = get_timestamp()
    last_user_events = None

    # query = "SELECT * FROM events WHERE event_type = 'view_item' ORDER BY customer_id"
    query = "SELECT * FROM events WHERE event_type = 'view_item' AND timestamp > '2019-03-01' ORDER BY customer_id"
    df = pd.read_sql(query, DB_CONNECTION_STRING)
    df = remove_unfrequent_items(df, 10)
    df = add_session_ids(df)
    df = remove_short_sessions(df)
    print(df)
    df.to_sql(
        f"preprocessed_events_month_{timestamp}",
        DB_CONNECTION_STRING,
        if_exists="append",
        index=False,
        chunksize=100000,
        method="multi",
    )
    # s = transform_to_sessions(df)
    # s.to_sql(f"click_sequences_{timestamp}", DB_CONNECTION_STRING, if_exists="append", index=False)


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


# sql_to_csv(
#     """
#         SELECT product_id, customer_id, timestamp, session_id, title, categories
#         FROM preprocessed_events_month_1603127051 e join products p on p.id = e.product_id
#         ORDER BY timestamp DESC LIMIT 1000000
#     """,
#     "data/events_month_1000000.csv",
# )
