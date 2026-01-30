from collections import defaultdict, Counter
from itertools import combinations
import math


def popularity_topk(train, k=20):
    pop = train.groupby("item_id")["weight"].sum().sort_values(ascending=False)
    return pop.head(k).index.tolist(), pop


def item_item_cosine(train, min_item_freq=5, max_items_per_user=50):
    user_items = (
        train.sort_values("ts")
        .groupby("user_id")["item_id"]
        .apply(lambda x: list(dict.fromkeys(x)))
    )

    item_freq = Counter()
    for items in user_items:
        item_freq.update(items)

    filtered_user_items = []
    for items in user_items:
        items_f = [i for i in items if item_freq[i] >= min_item_freq]
        if len(items_f) > max_items_per_user:
            items_f = items_f[-max_items_per_user:]
        if len(items_f) >= 2:
            filtered_user_items.append(items_f)

    co = defaultdict(Counter)
    for items in filtered_user_items:
        for i, j in combinations(items, 2):
            co[i][j] += 1
            co[j][i] += 1

    def recommend(user_id, user_items_map, k=20):
        items = user_items_map.get(user_id, [])
        scores = Counter()
        for it in items[-10:]:
            for j, c in co.get(it, {}).items():
                denom = math.sqrt(item_freq[it] * item_freq[j])
                if denom == 0:
                    continue
                scores[j] += c / denom
        seen = set(items)
        for s in list(scores.keys()):
            if s in seen:
                del scores[s]
        return [j for j, _ in scores.most_common(k)]

    return co, item_freq, recommend
