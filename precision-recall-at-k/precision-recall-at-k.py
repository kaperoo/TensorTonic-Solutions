def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    rel = 0
    for i in recommended[:k]:
        if i in relevant:
            rel += 1

    return [rel/k, rel/len(relevant)]