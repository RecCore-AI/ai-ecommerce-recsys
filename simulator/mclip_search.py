"""
M-CLIP 텍스트 인코더 (한/영 텍스트 → 이미지 검색용) — 서빙 측 로더

목적: 텍스트 쿼리를 CLIP 이미지 공간에 정렬된 다국어 텍스트 인코더로 임베딩해
      기존 image.index(CLIP ViT-B/32 이미지 임베딩)에서 '의미적으로 닮은' 상품을 회수한다.

왜 필요한가:
  - CLIP 의 text<->text 매칭은 글자가 거의 같아야만 통해 의미검색이 약하다.
  - M-CLIP(clip-ViT-B-32-multilingual-v1)은 50+개 언어 텍스트를 CLIP 이미지 공간에
    distillation 으로 정렬한 모델 → text→image 검색이 진짜 의미검색이 된다(영어 실측 우수).
  - 경량 M-CLIP 은 한국어가 약해, 한글은 ko_fashion 도메인 사전으로 핵심 토큰을 영어로
    치환한 뒤 인코딩한다(범용 MT 의 명사 환각 회피).

image.index 자체는 phase3 가 이미 로드하므로 여기서는 '쿼리 인코딩'만 담당한다(중복 로드 방지).
모든 모델은 로컬 동작 → 명세 line 403(외부 클라우드 API 금지) 준수.
"""
import numpy as np

import ko_fashion

MCLIP_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


class MclipEncoder:
    """프로세스당 1개. encode_query(query) -> 정규화된 [1,512] float32 벡터."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MCLIP_MODEL)

    def encode_query(self, query: str) -> np.ndarray:
        # 한글이면 패션 사전으로 영어 토큰화(원피스→dress …) 후 인코딩.
        text = ko_fashion.normalize_query(query)
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb.astype("float32")


_ENC = None


def get_encoder() -> MclipEncoder:
    global _ENC
    if _ENC is None:
        _ENC = MclipEncoder()
    return _ENC
