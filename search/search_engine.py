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
    def __init__(
        self,
        products_path: str = "simulator/data/products.csv",
        artifacts_dir: str = "search/artifacts",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        self.products_path = Path(products_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.vectorizer_path = self.artifacts_dir / "text_vectorizer.pkl"
        self.matrix_path = self.artifacts_dir / "baseline_matrix.npy"
        self.clip_embeddings_path = self.artifacts_dir / "clip_text_embeddings.npy"
        self.faiss_index_path = self.artifacts_dir / "clip_text_flat.index"
        self.faiss_ivfpq_index_path = self.artifacts_dir / "clip_text_ivfpq.index"

        self.faiss_index = None
        self.faiss_ivfpq_index = None
        self.faiss_ivfpq_nprobe = 8

        self.clip_model_name = clip_model_name
        self.device = device

        self.products: pd.DataFrame | None = None

        self.vectorizer: TfidfVectorizer | None = None
        self.product_matrix: np.ndarray | None = None

        self.clip_model = None
        self.clip_processor = None
        self.clip_text_embeddings: np.ndarray | None = None

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
        return (
            f"{row['category_L1']} "
            f"{row['category_L2']} "
            f"{row['category_L3']} "
            f"{row['price_tier']}"
        )

    def load_model(self, method: str = "tfidf", rebuild: bool = False) -> None:
        if self.products is None:
            self.load_products()

        if method == "tfidf":
            self._load_tfidf_model(rebuild=rebuild)
        elif method == "clip":
            self._load_clip_model()
            self._load_or_build_clip_embeddings(rebuild=rebuild)

        elif method == "clip_faiss":
            self._load_clip_model()
            self._load_or_build_clip_embeddings(rebuild=rebuild)
            self._load_or_build_faiss_index(rebuild=rebuild)

        elif method == "clip_faiss_ivfpq":
            self._load_clip_model()
            self._load_or_build_clip_embeddings(rebuild=rebuild)
            self._load_or_build_faiss_ivfpq_index(rebuild=rebuild)
        else:
            raise ValueError(f"지원하지 않는 method입니다: {method}")

    def _load_tfidf_model(self, rebuild: bool = False) -> None:
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

    def _load_clip_model(self) -> None:
        if self.clip_model is not None and self.clip_processor is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()

    def _load_or_build_clip_embeddings(
        self,
        rebuild: bool = False,
        batch_size: int = 32,
    ) -> None:
        assert self.products is not None

        if not rebuild and self.clip_embeddings_path.exists():
            self.clip_text_embeddings = np.load(self.clip_embeddings_path, allow_pickle=False)
            return

        texts = self.products["search_text"].astype(str).tolist()
        embeddings = []

        print(f"[CLIP] 상품 텍스트 {len(texts)}개 임베딩 생성 시작")

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_embeddings = self._encode_texts_clip(batch_texts)
            embeddings.append(batch_embeddings)

            end = min(start + batch_size, len(texts))
            print(f"[CLIP] {end}/{len(texts)} 완료")

        self.clip_text_embeddings = np.vstack(embeddings).astype(np.float32)
        np.save(self.clip_embeddings_path, self.clip_text_embeddings)

        print(f"[CLIP] 임베딩 저장 완료: {self.clip_embeddings_path}")

    def _load_or_build_faiss_index(self, rebuild: bool = False) -> None:
        if self.clip_text_embeddings is None:
            self._load_or_build_clip_embeddings(rebuild=False)

        assert self.clip_text_embeddings is not None

        import faiss

        if not rebuild and self.faiss_index_path.exists():
            self.faiss_index = faiss.read_index(str(self.faiss_index_path))
            return

        dimension = self.clip_text_embeddings.shape[1]

        # CLIP 임베딩은 이미 L2 normalize되어 있으므로
        # Inner Product 검색은 cosine similarity 검색과 같은 의미로 사용할 수 있다.
        index = faiss.IndexFlatIP(dimension)
        index.add(self.clip_text_embeddings.astype(np.float32))

        faiss.write_index(index, str(self.faiss_index_path))
        self.faiss_index = index

        print(f"[FAISS] IndexFlatIP 저장 완료: {self.faiss_index_path}")

    def _load_or_build_faiss_ivfpq_index(
        self,
        rebuild: bool = False,
        nlist: int = 100,
        m: int = 16,
        nbits: int = 8,
        nprobe: int = 8,
    ) -> None:
        if self.clip_text_embeddings is None:
            self._load_or_build_clip_embeddings(rebuild=False)

        assert self.clip_text_embeddings is not None

        import faiss

        self.faiss_ivfpq_nprobe = nprobe

        if not rebuild and self.faiss_ivfpq_index_path.exists():
            self.faiss_ivfpq_index = faiss.read_index(str(self.faiss_ivfpq_index_path))
            self.faiss_ivfpq_index.nprobe = nprobe
            return

        embeddings = self.clip_text_embeddings.astype(np.float32)
        num_vectors, dimension = embeddings.shape

        if dimension % m != 0:
            raise ValueError(
                f"IVFPQ의 m은 embedding dimension을 나누어떨어지게 해야 합니다. "
                f"dimension={dimension}, m={m}"
            )

        # 데이터가 너무 적으면 nlist를 자동으로 줄인다.
        nlist = min(nlist, max(1, int(np.sqrt(num_vectors))))

        # IVF의 coarse quantizer
        quantizer = faiss.IndexFlatIP(dimension)

        # CLIP embedding은 L2 normalize되어 있으므로 Inner Product는 cosine similarity처럼 사용 가능
        index = faiss.IndexIVFPQ(
            quantizer,
            dimension,
            nlist,
            m,
            nbits,
            faiss.METRIC_INNER_PRODUCT,
        )

        print(
            f"[FAISS-IVFPQ] 인덱스 학습 시작: "
            f"vectors={num_vectors}, dim={dimension}, nlist={nlist}, m={m}, nbits={nbits}"
        )

        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

        faiss.write_index(index, str(self.faiss_ivfpq_index_path))
        self.faiss_ivfpq_index = index

        print(f"[FAISS-IVFPQ] 인덱스 저장 완료: {self.faiss_ivfpq_index_path}")

    def _search_clip_faiss(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if self.faiss_index is None:
            self.load_model(method="clip_faiss", rebuild=False)

        assert self.products is not None
        assert self.faiss_index is not None

        top_k = min(top_k, len(self.products))

        query_embedding = self._encode_texts_clip([query_text]).astype(np.float32)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        return self._format_results(indices[0], scores[0])

    def _search_clip_faiss_ivfpq(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if self.faiss_ivfpq_index is None:
            self.load_model(method="clip_faiss_ivfpq", rebuild=False)

        assert self.products is not None
        assert self.faiss_ivfpq_index is not None

        top_k = min(top_k, len(self.products))

        query_embedding = self._encode_texts_clip([query_text]).astype(np.float32)

        scores, indices = self.faiss_ivfpq_index.search(query_embedding, top_k)

        return self._format_results(indices[0], scores[0])

    def _encode_texts_clip(self, texts: list[str]) -> np.ndarray:
        if self.clip_model is None or self.clip_processor is None:
            self._load_clip_model()

        assert self.clip_model is not None
        assert self.clip_processor is not None

        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask"]
        }

        with self.torch.no_grad():
            outputs = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

        # transformers 버전에 따라 Tensor가 아니라 ModelOutput이 나오는 경우 방어
        if self.torch.is_tensor(outputs):
            text_features = outputs
        elif hasattr(outputs, "text_embeds"):
            text_features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            text_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            text_features = outputs.last_hidden_state[:, 0, :]
        else:
            raise TypeError(f"지원하지 않는 CLIP 출력 타입입니다: {type(outputs)}")

        text_features = text_features.detach().cpu().numpy().astype(np.float32)
        text_features = self._l2_normalize(text_features)

        return text_features

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def search(
        self,
        query_text: str | None = None,
        top_k: int = 10,
        search_type: str = "text",
        method: str = "tfidf",
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        if not query_text or not query_text.strip():
            raise ValueError("query_text가 비어 있습니다.")

        if self.products is None:
            self.load_products()

        if method == "tfidf":
            results = self._search_tfidf(query_text=query_text, top_k=top_k)
        elif method == "clip":
            results = self._search_clip(query_text=query_text, top_k=top_k)
        elif method == "clip_faiss":
            results = self._search_clip_faiss(query_text=query_text, top_k=top_k)
        elif method == "clip_faiss_ivfpq":
            results = self._search_clip_faiss_ivfpq(query_text=query_text, top_k=top_k)
        else:
            raise ValueError(f"지원하지 않는 method입니다: {method}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "search_type": search_type,
            "method": method,
            "results": results,
            "latency_ms": round(latency_ms, 3),
            "total_count": len(results),
        }

    def _search_tfidf(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if self.vectorizer is None or self.product_matrix is None:
            self.load_model(method="tfidf", rebuild=False)

        assert self.products is not None
        assert self.vectorizer is not None
        assert self.product_matrix is not None

        top_k = min(top_k, len(self.products))

        query_vector = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        scores = cosine_similarity(query_vector, self.product_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return self._format_results(top_indices, scores)

    def _search_clip(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if self.clip_text_embeddings is None:
            self.load_model(method="clip", rebuild=False)

        assert self.products is not None
        assert self.clip_text_embeddings is not None

        top_k = min(top_k, len(self.products))

        query_embedding = self._encode_texts_clip([query_text])
        scores = query_embedding @ self.clip_text_embeddings.T
        scores = scores[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return self._format_results(top_indices, scores)

    def _format_results(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        assert self.products is not None

        results = []

        for rank_pos, idx in enumerate(indices):
            idx = int(idx)

            # FAISS에서 검색 실패 시 -1이 나올 수 있어서 방어
            if idx < 0 or idx >= len(self.products):
                continue

            row = self.products.iloc[idx]

            # 일반 numpy 검색: scores 길이 == 전체 상품 수
            # FAISS 검색: scores 길이 == top_k
            if len(scores) == len(indices):
                score = float(scores[rank_pos])
            else:
                score = float(scores[idx])

            results.append(
                {
                    "product_id": str(row["product_id"]),
                    "name": str(row["name"]),
                    "score": float(round(score, 6)),
                    "price": int(row["price"]),
                    "category_L1": str(row["category_L1"]),
                    "category_L2": str(row["category_L2"]),
                    "category_L3": str(row["category_L3"]),
                    "price_tier": str(row["price_tier"]),
                }
            )

        return results


if __name__ == "__main__":
    engine = SearchEngine()
    engine.load_products()

    response = engine.search("의류 상의 니트 low", top_k=10, method="clip")
    print(response)