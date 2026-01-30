import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

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
test  = events[events["ts"] > cutoff]

# 1) preparar matriz implicita
user_ids = train["user_id"].astype("category")
item_ids = train["item_id"].astype("category")

item_map = dict(enumerate(item_ids.cat.categories))

user_idx = user_ids.cat.codes
item_idx = item_ids.cat.codes

matrix = csr_matrix(
    (train["weight"], (user_idx, item_idx))
)

# 2) treinar ALS
als = AlternatingLeastSquares(
    factors=64,
    regularization=0.01,
    iterations=15,
    random_state=42
)

als.fit(matrix)

# 3) funcao de recomendacao
def recommend_als(user_id, k=20):
    if user_id not in user_ids.cat.categories:
        return []

    uidx = user_ids.cat.categories.get_loc(user_id)
    recs, _ = als.recommend(
        uidx, matrix[uidx], N=k, filter_already_liked_items=True
    )
    return [item_map[i] for i in recs]

# 4) avaliacao recall@20

def recall_at_k_als(test, rec_fn, k=20, max_users=2000):
    hits = 0
    total = 0

    for idx, (user, group) in enumerate(test.groupby("user_id")):
        if idx >= max_users:
            break
        true_items = set(group["item_id"])
        rec_items = set(rec_fn(user, k))

        hits += len(true_items & rec_items)
        total += len(true_items)

    return hits / total if total else 0

recall_als = recall_at_k_als(test, recommend_als, k=20, max_users=2000)
print("ALS Recall@20 (sample 2000 users):", recall_als)
