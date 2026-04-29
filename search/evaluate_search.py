from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from search_engine import SearchEngine


def reciprocal_rank(results: list[dict], relevant_product_id: str) -> float:
    for rank, item in enumerate(results, start=1):
        if item["product_id"] == relevant_product_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[dict], relevant_product_id: str, k: int = 10) -> float:
    for rank, item in enumerate(results[:k], start=1):
        if item["product_id"] == relevant_product_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def make_query_from_product(row: pd.Series) -> str:
    return (
        f"{row['category_L1']} "
        f"{row['category_L2']} "
        f"{row['category_L3']} "
        f"{row['price_tier']}"
    )


def main() -> None:
    products_path = Path("simulator/data/products.csv")
    test_logs_path = Path("simulator/data/test_logs.csv")

    if not products_path.exists():
        raise FileNotFoundError(f"products.csv 없음: {products_path}")
    if not test_logs_path.exists():
        raise FileNotFoundError(f"test_logs.csv 없음: {test_logs_path}")

    products = pd.read_csv(products_path)
    test_logs = pd.read_csv(test_logs_path)

    search_logs = test_logs[test_logs["event_type"] == "search"].copy()

    if search_logs.empty:
        raise ValueError("test_logs.csv에 event_type=search 로그가 없습니다.")

    product_map = products.set_index("product_id")

    engine = SearchEngine(
        products_path=str(products_path),
        artifacts_dir="search/artifacts",
    )
    engine.load_products()
    engine.load_model(rebuild=False)

    mrr_scores = []
    ndcg_scores = []
    latency_list = []

    # 너무 많으면 오래 걸릴 수 있으므로 처음에는 1000개만 평가
    max_eval = min(1000, len(search_logs))
    eval_logs = search_logs.head(max_eval)

    valid_count = 0

    for _, log in eval_logs.iterrows():
        product_id = log["product_id"]

        if product_id not in product_map.index:
            continue

        product_row = product_map.loc[product_id]
        query = make_query_from_product(product_row)

        response = engine.search(query_text=query, top_k=10)
        results = response["results"]

        mrr_scores.append(reciprocal_rank(results, product_id))
        ndcg_scores.append(ndcg_at_k(results, product_id, k=10))
        latency_list.append(response["latency_ms"])
        valid_count += 1

    if valid_count == 0:
        raise ValueError("평가 가능한 search 로그가 없습니다.")

    mrr = sum(mrr_scores) / len(mrr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_latency = sum(latency_list) / len(latency_list)
    p95_latency = sorted(latency_list)[int(len(latency_list) * 0.95) - 1]

    print("\n========== Search Baseline Evaluation ==========")
    print(f"eval_count: {valid_count}")
    print(f"MRR: {mrr:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")
    print(f"avg_latency_ms: {avg_latency:.3f}")
    print(f"p95_latency_ms: {p95_latency:.3f}")


if __name__ == "__main__":
    main()