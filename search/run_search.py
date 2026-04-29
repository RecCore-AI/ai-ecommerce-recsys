from search_engine import SearchEngine


def print_results(response: dict) -> None:
    print("\n========== 검색 결과 ==========")
    print(f"search_type: {response['search_type']}")
    print(f"method: {response['method']}")
    print(f"latency_ms: {response['latency_ms']}")
    print(f"total_count: {response['total_count']}")

    for i, item in enumerate(response["results"], start=1):
        print(
            f"{i}. {item['product_id']} | "
            f"{item['name']} | "
            f"price={item['price']} | "
            f"tier={item['price_tier']} | "
            f"score={item['score']}"
        )


def run_method(engine: SearchEngine, method: str, queries: list[str]) -> None:
    title = method.upper()
    print(f"\n\n==================== {title} SEARCH ====================")

    engine.load_model(method=method, rebuild=False)

    for query in queries:
        print(f"\n\nQuery: {query}")
        response = engine.search(query_text=query, top_k=10, method=method)
        print_results(response)


def main() -> None:
    engine = SearchEngine(
        products_path="simulator/data/products.csv",
        artifacts_dir="search/artifacts",
        device="cpu",
    )

    engine.load_products()

    queries = [
        "의류 상의 니트 low",
        "전자제품 PC 키보드 low",
        "전자제품 모바일 스마트폰 high",
        "의류 하의 청바지 medium",
        "전자제품 PC 모니터 high",
    ]

    run_method(engine, "tfidf", queries)
    run_method(engine, "clip", queries)
    run_method(engine, "clip_faiss", queries)


if __name__ == "__main__":
    main()