def recall_at_k(test, recommendations, k=20):
    hits = 0
    total = 0
    for user, group in test.groupby("user_id"):
        true_items = set(group["item_id"])
        rec_items = set(recommendations[:k])
        hits += len(true_items & rec_items)
        total += len(true_items)
    return hits / total if total else 0


def recall_at_k_personalized(test, rec_fn, k=20, max_users=2000):
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
