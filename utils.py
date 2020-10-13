from datetime import datetime

def get_timestamp():
    return int(datetime.now(tz=None).timestamp())