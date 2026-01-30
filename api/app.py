from fastapi import FastAPI
import os
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb

app = FastAPI()

TOP_K = 20
ALS_K = 100
DATA_PATH = os.path.join("data", "events.csv")
ARTIFACTS_DIR = "artifacts"


def _load_events():
    events = pd.read_csv(DATA_PATH)
    weights = {
        "view": 1,
        "addtocart": 3,
        "transaction": 8,
    }
    events["weight"] = events["event"].map(weights)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    events = events[["visitorid", "itemid", "timestamp", "weight"]]
    events.columns = ["user_id", "item_id", "ts", "weight"]
    return events


def _load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    als = AlternatingLeastSquares().load(os.path.join(ARTIFACTS_DIR, "als_model.npz"))
    ranker = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, "ranker.txt"))
    return mappings, stats, als, ranker


def _build_matrix(train, mappings):
    user_ids = pd.Categorical(train["user_id"], categories=mappings["user_categories"])
    item_ids = pd.Categorical(train["item_id"], categories=mappings["item_categories"])
    matrix = csr_matrix((train["weight"], (user_ids.codes, item_ids.codes)))
    return matrix, user_ids, item_ids


def _load_state():
    events = _load_events()
    cutoff = events["ts"].quantile(0.8)
    train = events[events["ts"] <= cutoff]
    mappings, stats, als, ranker = _load_artifacts()
    matrix, user_ids, item_ids = _build_matrix(train, mappings)
    item_map = dict(enumerate(item_ids.categories))
    return {
        "item_pop": stats["item_pop"],
        "pop_7d": stats["pop_7d"],
        "ui_cnt": stats["ui_cnt"],
        "top_items": stats["top_items"],
        "feat_cols": stats["feat_cols"],
        "user_ids": user_ids,
        "item_map": item_map,
        "matrix": matrix,
        "als": als,
        "ranker": ranker,
    }


STATE = _load_state()


def candidates_als(user_id, k=ALS_K):
    user_ids = STATE["user_ids"]
    if user_id not in user_ids.categories:
        return []
    uidx = user_ids.categories.get_loc(user_id)
    recs, _ = STATE["als"].recommend(
        uidx, STATE["matrix"][uidx], N=k, filter_already_liked_items=True
    )
    return [STATE["item_map"][i] for i in recs]


def fallback_popular(k=TOP_K):
    return STATE["top_items"][:k]


def build_features(user_id, item_ids):
    df = pd.DataFrame({"user_id": user_id, "item_id": item_ids})
    df["item_pop"] = df["item_id"].map(STATE["item_pop"]).fillna(0)
    df["item_pop_7d"] = df["item_id"].map(STATE["pop_7d"]).fillna(0)

    idx = list(zip([user_id] * len(item_ids), item_ids))
    df["ui_cnt"] = [STATE["ui_cnt"].get(x, 0) for x in idx]
    return df


@app.get("/recommend")
def recommend(user_id: int):
    cands = candidates_als(user_id, k=ALS_K)
    if not cands:
        return {"user_id": user_id, "items": fallback_popular(TOP_K), "source": "fallback_pop"}

    feat_df = build_features(user_id, cands)
    feat_df["score"] = STATE["ranker"].predict(feat_df[STATE["feat_cols"]])
    recs = (
        feat_df.sort_values("score", ascending=False)
        .head(TOP_K)["item_id"]
        .tolist()
    )
    return {"user_id": user_id, "items": recs, "source": "als+lgbm"}
