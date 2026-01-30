import pandas as pd
import lightgbm as lgb


def build_rank_data(train, test, als, matrix, user_ids, item_ids, k=100, max_users=2000):
    item_map = dict(enumerate(item_ids.cat.categories))
    rows = []
    for idx, (user, group) in enumerate(test.groupby("user_id")):
        if idx >= max_users:
            break
        if user not in user_ids.cat.categories:
            continue
        uidx = user_ids.cat.categories.get_loc(user)
        recs, _ = als.recommend(
            uidx, matrix[uidx], N=k, filter_already_liked_items=True
        )
        cands = [item_map[i] for i in recs]
        true = set(group["item_id"])
        for it in cands:
            rows.append(
                {"user_id": user, "item_id": it, "label": 1 if it in true else 0}
            )
    return pd.DataFrame(rows)


def add_features(rank_df, train):
    item_pop = train.groupby("item_id")["weight"].sum()
    cut_train = train["ts"].max() - pd.Timedelta(days=7)
    pop_7d = train[train["ts"] >= cut_train].groupby("item_id")["weight"].sum()
    ui_cnt = train.groupby(["user_id", "item_id"]).size()

    rank_df["item_pop"] = rank_df["item_id"].map(item_pop).fillna(0)
    rank_df["item_pop_7d"] = rank_df["item_id"].map(pop_7d).fillna(0)
    rank_df["ui_cnt"] = (
        rank_df.set_index(["user_id", "item_id"]).index.map(ui_cnt).fillna(0)
    )

    feat_cols = ["item_pop", "item_pop_7d", "ui_cnt"]
    stats = {
        "item_pop": item_pop,
        "pop_7d": pop_7d,
        "ui_cnt": ui_cnt.to_dict(),
        "feat_cols": feat_cols,
    }
    return rank_df, feat_cols, stats


def train_ranker(rank_df, feat_cols):
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
    return ranker
