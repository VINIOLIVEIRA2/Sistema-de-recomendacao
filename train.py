import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb

DATA_PATH = os.path.join("data", "events.csv")
ARTIFACTS_DIR = "artifacts"
ALS_K = 100
TOP_K = 20
MAX_USERS_RANK_TRAIN = 2000


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


def train_als(train):
    user_ids = train["user_id"].astype("category")
    item_ids = train["item_id"].astype("category")
    user_idx = user_ids.cat.codes
    item_idx = item_ids.cat.codes
    matrix = csr_matrix((train["weight"], (user_idx, item_idx)))

    als = AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=15,
        random_state=42,
    )
    als.fit(matrix)
    return als, matrix, user_ids, item_ids


def build_rank_data(train, test, als, matrix, user_ids, item_ids):
    item_map = dict(enumerate(item_ids.cat.categories))
    rows = []
    for idx, (user, group) in enumerate(test.groupby("user_id")):
        if idx >= MAX_USERS_RANK_TRAIN:
            break
        if user not in user_ids.cat.categories:
            continue
        uidx = user_ids.cat.categories.get_loc(user)
        recs, _ = als.recommend(
            uidx, matrix[uidx], N=ALS_K, filter_already_liked_items=True
        )
        cands = [item_map[i] for i in recs]
        true = set(group["item_id"])
        for it in cands:
            rows.append(
                {"user_id": user, "item_id": it, "label": 1 if it in true else 0}
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    events = load_events(DATA_PATH)
    train, test = split_temporal(events)

    # baselines
    item_pop = train.groupby("item_id")["weight"].sum()
    top_items = item_pop.sort_values(ascending=False).head(200).index.tolist()

    # ALS
    als, matrix, user_ids, item_ids = train_als(train)

    # ranking data
    rank_df = build_rank_data(train, test, als, matrix, user_ids, item_ids)

    # features
    cut_train = train["ts"].max() - pd.Timedelta(days=7)
    pop_7d = train[train["ts"] >= cut_train].groupby("item_id")["weight"].sum()
    ui_cnt = train.groupby(["user_id", "item_id"]).size()

    rank_df["item_pop"] = rank_df["item_id"].map(item_pop).fillna(0)
    rank_df["item_pop_7d"] = rank_df["item_id"].map(pop_7d).fillna(0)
    rank_df["ui_cnt"] = (
        rank_df.set_index(["user_id", "item_id"]).index.map(ui_cnt).fillna(0)
    )

    feat_cols = ["item_pop", "item_pop_7d", "ui_cnt"]
    X = rank_df[feat_cols]
    y = rank_df["label"]
    groups = rank_df.groupby("user_id").size().values

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    ranker.fit(X, y, group=groups)

    # save artifacts
    als.save(os.path.join(ARTIFACTS_DIR, "als_model.npz"))
    ranker.booster_.save_model(os.path.join(ARTIFACTS_DIR, "ranker.txt"))

    mappings = {
        "user_categories": list(user_ids.cat.categories),
        "item_categories": list(item_ids.cat.categories),
    }
    with open(os.path.join(ARTIFACTS_DIR, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)

    stats = {
        "item_pop": item_pop,
        "pop_7d": pop_7d,
        "ui_cnt": ui_cnt.to_dict(),
        "top_items": top_items,
        "feat_cols": feat_cols,
    }
    with open(os.path.join(ARTIFACTS_DIR, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    print("artifacts saved")


if __name__ == "__main__":
    main()
