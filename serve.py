import os
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb

ARTIFACTS_DIR = "artifacts"
DATA_PATH = os.path.join("data", "events.csv")
ALS_K = 100
TOP_K = 20


def load_events(path):
    events = pd.read_csv(path)
    weights = {"view": 1, "addtocart": 3, "transaction": 8}
    events["weight"] = events["event"].map(weights)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    events = events[["visitorid", "itemid", "timestamp", "weight"]]
    events.columns = ["user_id", "item_id", "ts", "weight"]
    return events


def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    als = AlternatingLeastSquares()
    als = als.load(os.path.join(ARTIFACTS_DIR, "als_model.npz"))

    booster = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, "ranker.txt"))

    return mappings, stats, als, booster


def build_matrix(train, mappings):
    user_ids = pd.Categorical(train["user_id"], categories=mappings["user_categories"])
    item_ids = pd.Categorical(train["item_id"], categories=mappings["item_categories"])
    user_idx = user_ids.codes
    item_idx = item_ids.codes
    matrix = csr_matrix((train["weight"], (user_idx, item_idx)))
    return matrix, user_ids, item_ids


def build_features(user_id, item_ids, stats):
    df = pd.DataFrame({"user_id": user_id, "item_id": item_ids})
    df["item_pop"] = df["item_id"].map(stats["item_pop"]).fillna(0)
    df["item_pop_7d"] = df["item_id"].map(stats["pop_7d"]).fillna(0)
    idx = list(zip([user_id] * len(item_ids), item_ids))
    df["ui_cnt"] = [stats["ui_cnt"].get(x, 0) for x in idx]
    return df


def recommend(user_id, als, matrix, user_ids, item_map, stats, ranker, feat_cols):
    if user_id not in user_ids.categories:
        return stats["top_items"][:TOP_K], "fallback_pop"

    uidx = user_ids.categories.get_loc(user_id)
    recs, _ = als.recommend(uidx, matrix[uidx], N=ALS_K, filter_already_liked_items=True)
    cands = [item_map[i] for i in recs]

    if not cands:
        return stats["top_items"][:TOP_K], "fallback_pop"

    feat_df = build_features(user_id, cands, stats)
    feat_df["score"] = ranker.predict(feat_df[feat_cols])
    recs = (
        feat_df.sort_values("score", ascending=False)
        .head(TOP_K)["item_id"]
        .tolist()
    )
    return recs, "als+lgbm"


def main():
    events = load_events(DATA_PATH)
    cutoff = events["ts"].quantile(0.8)
    train = events[events["ts"] <= cutoff]

    mappings, stats, als, booster = load_artifacts()
    matrix, user_ids, item_ids = build_matrix(train, mappings)
    item_map = dict(enumerate(item_ids.categories))
    feat_cols = stats["feat_cols"]

    items, source = recommend(123, als, matrix, user_ids, item_map, stats, booster, feat_cols)
    print({"user_id": 123, "items": items, "source": source})


if __name__ == "__main__":
    main()
