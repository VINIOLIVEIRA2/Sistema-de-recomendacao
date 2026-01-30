import pandas as pd


def load_events(path):
    events = pd.read_csv(path)
    weights = {"view": 1, "addtocart": 3, "transaction": 8}
    events["weight"] = events["event"].map(weights)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    events = events[["visitorid", "itemid", "timestamp", "weight"]]
    events.columns = ["user_id", "item_id", "ts", "weight"]
    return events


def split_temporal(events, q=0.8):
    cutoff = events["ts"].quantile(q)
    train = events[events["ts"] <= cutoff]
    test = events[events["ts"] > cutoff]
    return train, test
