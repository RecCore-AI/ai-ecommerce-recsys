from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    """
    현재 버전:
    - products.csv의 category_L1, category_L2, category_L3, price_tier, price를 이용한 TF-IDF 검색 baseline
    - 나중에 CLIP + FAISS로 교체하기 쉽게 load_model(), search() 인터페이스 유지
    """

    def __init__(
        self,
        products_path: str = "simulator/data/products.csv",
        artifacts_dir: str = "search/artifacts",
    ) -> None:
        self.products_path = Path(products_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.vectorizer_path = self.artifacts_dir / "text_vectorizer.pkl"
        self.matrix_path = self.artifacts_dir / "baseline_matrix.npy"

        self.products: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.product_matrix: np.ndarray | None = None

    def load_products(self) -> None:
        if not self.products_path.exists():
            raise FileNotFoundError(f"products.csv를 찾을 수 없습니다: {self.products_path}")

        products = pd.read_csv(self.products_path)

        required_columns = {
            "product_id",
            "category_L1",
            "category_L2",
            "category_L3",
            "price",
            "price_tier",
        }
        missing = required_columns - set(products.columns)
        if missing:
            raise ValueError(f"products.csv에 필요한 컬럼이 없습니다: {missing}")

        products["name"] = products.apply(self._make_product_name, axis=1)
        products["search_text"] = products.apply(self._make_search_text, axis=1)

        self.products = products

    def _make_product_name(self, row: pd.Series) -> str:
        return f"{row['category_L1']} {row['category_L2']} {row['category_L3']}"

    def _make_search_text(self, row: pd.Series) -> str:
        """
        현재 products.csv에는 상품명/설명/이미지가 없으므로
        카테고리와 가격대를 검색 텍스트로 사용한다.
        """
        return (
            f"{row['category_L1']} "
            f"{row['category_L2']} "
            f"{row['category_L3']} "
            f"{row['price_tier']} "
            f"{row['price']}원"
        )

    def load_model(self, rebuild: bool = False) -> None:
        """
        현재 baseline에서는 '모델'이 TF-IDF vectorizer에 해당한다.
        나중에 CLIP으로 바꿀 때도 serving 쪽에서는 load_model()만 호출하면 되게 유지한다.
        """
        if self.products is None:
            self.load_products()

        assert self.products is not None

        if (
            not rebuild
            and self.vectorizer_path.exists()
            and self.matrix_path.exists()
        ):
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.product_matrix = np.load(self.matrix_path, allow_pickle=False)
            return

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
        )

        matrix = self.vectorizer.fit_transform(self.products["search_text"].astype(str))
        self.product_matrix = matrix.toarray().astype(np.float32)

        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        np.save(self.matrix_path, self.product_matrix)

    def search(
        self,
        query_text: str | None = None,
        top_k: int = 10,
        search_type: str = "text",
    ) -> dict[str, Any]:
        """
        /api/search에 붙이기 쉬운 형태의 핵심 검색 함수.
        현재는 text search만 지원한다.
        """
        start = time.perf_counter()

        if not query_text or not query_text.strip():
            raise ValueError("query_text가 비어 있습니다.")

        if self.products is None:
            self.load_products()

        if self.vectorizer is None or self.product_matrix is None:
            self.load_model()

        assert self.products is not None
        assert self.vectorizer is not None
        assert self.product_matrix is not None

        top_k = min(top_k, len(self.products))

        query_vector = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        scores = cosine_similarity(query_vector, self.product_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.products.iloc[idx]
            results.append(
                {
                    "product_id": str(row["product_id"]),
                    "name": str(row["name"]),
                    "score": float(round(scores[idx], 6)),
                    "price": int(row["price"]),
                    "category_L1": str(row["category_L1"]),
                    "category_L2": str(row["category_L2"]),
                    "category_L3": str(row["category_L3"]),
                    "price_tier": str(row["price_tier"]),
                }
            )

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "search_type": search_type,
            "results": results,
            "latency_ms": round(latency_ms, 3),
            "total_count": len(results),
        }


if __name__ == "__main__":
    engine = SearchEngine()
    engine.load_products()
    engine.load_model(rebuild=True)

    response = engine.search("의류 상의 니트 low", top_k=10)
    print(response)