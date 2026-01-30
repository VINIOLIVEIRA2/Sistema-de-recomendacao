import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb
from sklearn.metrics import ndcg_score

# carregar dataset
events = pd.read_csv("archive/events.csv")

weights = {
    "view": 1,
    "addtocart": 3,
    "transaction": 8
}

events["weight"] = events["event"].map(weights)
events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")

events = events[["visitorid", "itemid", "timestamp", "weight"]]
events.columns = ["user_id", "item_id", "ts", "weight"]

# split temporal
cutoff = events["ts"].quantile(0.8)
train = events[events["ts"] <= cutoff]
test = events[events["ts"] > cutoff]

# preparar matriz implicita
user_ids = train["user_id"].astype("category")
item_ids = train["item_id"].astype("category")

item_map = dict(enumerate(item_ids.cat.categories))

user_idx = user_ids.cat.codes
item_idx = item_ids.cat.codes

matrix = csr_matrix(
    (train["weight"], (user_idx, item_idx))
)

# treinar ALS
als = AlternatingLeastSquares(
    factors=64,
    regularization=0.01,
    iterations=15,
    random_state=42
)
als.fit(matrix)

# candidatos ALS
def candidates_als(user_id, k=100):
    if user_id not in user_ids.cat.categories:
        return []
    uidx = user_ids.cat.categories.get_loc(user_id)
    recs, _ = als.recommend(uidx, matrix[uidx], N=k, filter_already_liked_items=True)
    return [item_map[i] for i in recs]

# montar dataset de ranking (amostra)
MAX_USERS = 2000
rows = []
for idx, (user, group) in enumerate(test.groupby("user_id")):
    if idx >= MAX_USERS:
        break
    cands = candidates_als(user, k=100)
    true = set(group["item_id"])
    for it in cands:
        rows.append({
            "user_id": user,
            "item_id": it,
            "label": 1 if it in true else 0
        })

rank_df = pd.DataFrame(rows)
print(rank_df.head())
print("label mean:", rank_df["label"].mean())

# features simples
item_pop = train.groupby("item_id")["weight"].sum()
rank_df["item_pop"] = rank_df["item_id"].map(item_pop).fillna(0)

# features melhores
cut_train = train["ts"].max() - pd.Timedelta(days=7)
pop_7d = train[train["ts"] >= cut_train].groupby("item_id")["weight"].sum()
rank_df["item_pop_7d"] = rank_df["item_id"].map(pop_7d).fillna(0)

ui_cnt = train.groupby(["user_id", "item_id"]).size()
rank_df["ui_cnt"] = rank_df.set_index(["user_id", "item_id"]).index.map(ui_cnt).fillna(0)

# treinar LightGBM Ranker
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
    random_state=42
)

ranker.fit(X, y, group=groups)

# avaliar NDCG@20
def ndcg_at_20(df):
    ndcgs = []
    for _, g in df.groupby("user_id"):
        y_true = g["label"].values.reshape(1, -1)
        y_score = ranker.predict(g[feat_cols]).reshape(1, -1)
        ndcgs.append(ndcg_score(y_true, y_score, k=20))
    return np.mean(ndcgs)

print("NDCG@20:", ndcg_at_20(rank_df))
print("feat_cols:", feat_cols)

# avaliar Recall@20 pos-ranking
def recall_at_20_post_ranking(rank_df, ranker, feat_cols, k=20):
    hits = 0
    total = 0

    for user, g in rank_df.groupby("user_id"):
        g = g.copy()
        g["score"] = ranker.predict(g[feat_cols])
        topk = set(g.sort_values("score", ascending=False).head(k)["item_id"])

        true_items = set(test[test["user_id"] == user]["item_id"])
        hits += len(true_items & topk)
        total += len(true_items)

    return hits / total if total else 0

recall_rank = recall_at_20_post_ranking(rank_df, ranker, feat_cols, k=20)
print("Recall@20 pos-ranking:", recall_rank)
