from search_engine import SearchEngine


def print_results(response: dict) -> None:
    print("\n========== 검색 결과 ==========")
    print(f"search_type: {response['search_type']}")
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


def main() -> None:
    engine = SearchEngine(
        products_path="simulator/data/products.csv",
        artifacts_dir="search/artifacts",
    )

    engine.load_products()
    engine.load_model(rebuild=True)

    queries = [
        "의류 상의 니트 low",
        "전자제품 PC 키보드 low",
        "전자제품 모바일 스마트폰 high",
        "의류 하의 청바지 medium",
        "전자제품 PC 모니터 high",
    ]

    for query in queries:
        print(f"\n\nQuery: {query}")
        response = engine.search(query_text=query, top_k=10)
        print_results(response)


if __name__ == "__main__":
    main()