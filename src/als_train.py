from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


def train_als(train, factors=64, regularization=0.01, iterations=15, random_state=42):
    user_ids = train["user_id"].astype("category")
    item_ids = train["item_id"].astype("category")
    user_idx = user_ids.cat.codes
    item_idx = item_ids.cat.codes
    matrix = csr_matrix((train["weight"], (user_idx, item_idx)))

    als = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    als.fit(matrix)
    return als, matrix, user_ids, item_ids
