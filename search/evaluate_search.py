from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from search_engine import SearchEngine


def exact_mrr(results: list[dict], relevant_product_id: str) -> float:
    for rank, item in enumerate(results, start=1):
        if item["product_id"] == relevant_product_id:
            return 1.0 / rank
    return 0.0


def exact_ndcg_at_k(results: list[dict], relevant_product_id: str, k: int = 10) -> float:
    for rank, item in enumerate(results[:k], start=1):
        if item["product_id"] == relevant_product_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def same_group(item: dict, target: pd.Series) -> bool:
    return (
        item["category_L1"] == target["category_L1"]
        and item["category_L2"] == target["category_L2"]
        and item["category_L3"] == target["category_L3"]
        and item["price_tier"] == target["price_tier"]
    )


def category_hit_at_k(results: list[dict], target: pd.Series, k: int = 10) -> float:
    for item in results[:k]:
        if same_group(item, target):
            return 1.0
    return 0.0


def category_mrr(results: list[dict], target: pd.Series) -> float:
    for rank, item in enumerate(results, start=1):
        if same_group(item, target):
            return 1.0 / rank
    return 0.0


def category_ndcg_at_k(results: list[dict], target: pd.Series, k: int = 10) -> float:
    dcg = 0.0

    for rank, item in enumerate(results[:k], start=1):
        if same_group(item, target):
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, k + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def make_query_from_product(row: pd.Series) -> str:
    return (
        f"{row['category_L1']} "
        f"{row['category_L2']} "
        f"{row['category_L3']} "
        f"{row['price_tier']}"
    )


def evaluate_method(
    engine: SearchEngine,
    products: pd.DataFrame,
    search_logs: pd.DataFrame,
    method: str,
    max_eval: int = 1000,
) -> None:
    product_map = products.set_index("product_id")

    exact_mrr_scores = []
    exact_ndcg_scores = []

    category_hit_scores = []
    category_mrr_scores = []
    category_ndcg_scores = []

    latency_list = []

    eval_logs = search_logs.head(min(max_eval, len(search_logs)))
    valid_count = 0

    engine.load_model(method=method, rebuild=False)

    for _, log in eval_logs.iterrows():
        product_id = log["product_id"]

        if product_id not in product_map.index:
            continue

        target_product = product_map.loc[product_id]
        query = make_query_from_product(target_product)

        response = engine.search(query_text=query, top_k=10, method=method)
        results = response["results"]

        exact_mrr_scores.append(exact_mrr(results, product_id))
        exact_ndcg_scores.append(exact_ndcg_at_k(results, product_id, k=10))

        category_hit_scores.append(category_hit_at_k(results, target_product, k=10))
        category_mrr_scores.append(category_mrr(results, target_product))
        category_ndcg_scores.append(category_ndcg_at_k(results, target_product, k=10))

        latency_list.append(response["latency_ms"])
        valid_count += 1

    if valid_count == 0:
        raise ValueError("평가 가능한 search 로그가 없습니다.")

    avg_latency = sum(latency_list) / len(latency_list)
    p95_latency = sorted(latency_list)[int(len(latency_list) * 0.95) - 1]

    print(f"\n\n========== {method.upper()} Search Evaluation ==========")
    print(f"eval_count: {valid_count}")

    print("\n[Exact Product ID Evaluation]")
    print(f"MRR: {sum(exact_mrr_scores) / len(exact_mrr_scores):.4f}")
    print(f"NDCG@10: {sum(exact_ndcg_scores) / len(exact_ndcg_scores):.4f}")

    print("\n[Category/Price-Tier Evaluation]")
    print(f"Category HitRate@10: {sum(category_hit_scores) / len(category_hit_scores):.4f}")
    print(f"Category MRR: {sum(category_mrr_scores) / len(category_mrr_scores):.4f}")
    print(f"Category NDCG@10: {sum(category_ndcg_scores) / len(category_ndcg_scores):.4f}")

    print("\n[Latency]")
    print(f"avg_latency_ms: {avg_latency:.3f}")
    print(f"p95_latency_ms: {p95_latency:.3f}")


def main() -> None:
    products_path = Path("simulator/data/products.csv")
    test_logs_path = Path("simulator/data/test_logs.csv")

    products = pd.read_csv(products_path)
    test_logs = pd.read_csv(test_logs_path)

    search_logs = test_logs[test_logs["event_type"] == "search"].copy()

    if search_logs.empty:
        raise ValueError("test_logs.csv에 event_type=search 로그가 없습니다.")

    engine = SearchEngine(
        products_path=str(products_path),
        artifacts_dir="search/artifacts",
        device="cpu",
    )
    engine.load_products()

    evaluate_method(
        engine=engine,
        products=products,
        search_logs=search_logs,
        method="tfidf",
        max_eval=1000,
    )

    evaluate_method(
        engine=engine,
        products=products,
        search_logs=search_logs,
        method="clip",
        max_eval=1000,
    )

    evaluate_method(
        engine=engine,
        products=products,
        search_logs=search_logs,
        method="clip_faiss",
        max_eval=1000,
    )


if __name__ == "__main__":
    main()