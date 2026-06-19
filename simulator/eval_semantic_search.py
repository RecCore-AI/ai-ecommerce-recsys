"""
의미/한국어 검색 평가 (baseline CLIP text.index  vs  배포 fused: CLIP text.index + M-CLIP)

기존 phase4 의 MRR/NDCG 는 '상품명 그대로'를 쿼리로 써서 정확일치만 측정한다
(MRR 0.92 가 높아 보이지만 의미·한국어 검색 능력은 측정조차 안 됨).

여기서는 '색상+품목' 의미 쿼리(한/영)를 만들고, 정답을 상품명 메타데이터로 자동 정의해
(상품명에 색상 AND 품목 동의어 포함 시 relevant) Precision@10 을 비교한다. 재현 가능, random 없음.

측정 경로(배포 phase3 와 동일):
  - baseline : CLIP Text Encoder(쿼리) → text.index
  - fused    : 영어 = CLIP text.index + M-CLIP(text→image) RRF / 한국어 = M-CLIP 전담
               (한글은 CLIP text↔text 가 노이즈라 제외)
"""
import json

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

import ko_fashion

PRODUCTS    = "data/products.csv"
TEXT_INDEX  = "data/indices/text.index"
IMAGE_INDEX = "data/indices/image.index"
TOPK        = 10
RRF_K       = 60
OUT         = "data/semantic_search_metrics.json"

QUERY_SET = [
    ("black hoodie",     "검정 후드티",   "black", ["hood"]),
    ("red dress",        "빨간 원피스",   "red",   ["dress"]),
    ("blue jeans",       "파란 청바지",   "blue",  ["jeans", "denim", "jegging"]),
    ("white sneakers",   "흰색 운동화",   "white", ["sneaker", "trainer"]),
    ("knitted sweater",  "니트 스웨터",   "",      ["knit", "jumper", "sweater"]),
    ("leather bag",      "가죽 가방",     "",      ["bag", "shopper", "tote"]),
    ("pink skirt",       "분홍 치마",     "pink",  ["skirt"]),
    ("grey cardigan",    "회색 가디건",   "grey",  ["cardigan"]),
    ("blue shirt",       "파란 셔츠",     "blue",  ["shirt"]),
    ("black jacket",     "검정 자켓",     "black", ["jacket", "blazer"]),
    ("green t-shirt",    "초록 티셔츠",   "green", ["t-shirt", "tee", " top"]),
    ("beige coat",       "베이지 코트",   "beige", ["coat"]),
]


def is_relevant(name, color, items):
    n = str(name).lower()
    if color and color not in n:
        return False
    return any(s in n for s in items)


def precision_at_k(idxs, names, color, items):
    if len(idxs) == 0:
        return 0.0
    return sum(is_relevant(names[int(i)], color, items) for i in idxs if int(i) >= 0) / len(idxs)


def main():
    df = pd.read_csv(PRODUCTS)
    names = df["product_name"].fillna("").tolist()
    text_idx  = faiss.read_index(TEXT_INDEX)
    image_idx = faiss.read_index(IMAGE_INDEX)

    print("[load] CLIP (baseline) + M-CLIP (신규)")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    mclip = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

    def clip_rank(q, pool):
        tok = clip_proc.tokenizer(ko_fashion.strip_price(q), return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            tf = clip_model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            if not isinstance(tf, torch.Tensor):
                pooler = getattr(tf, "pooler_output", None)
                tf = pooler if pooler is not None else getattr(tf, "text_embeds", None)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        _, I = text_idx.search(tf.cpu().numpy().astype("float32"), pool)
        return [int(i) for i in I[0] if int(i) >= 0]

    def mclip_rank(q, pool):
        v = mclip.encode([ko_fashion.normalize_query(ko_fashion.strip_price(q))], normalize_embeddings=True).astype("float32")
        _, I = image_idx.search(v, pool)
        return [int(i) for i in I[0] if int(i) >= 0]

    def fused_rank(q):
        """배포 phase3 와 동일: 영어=CLIP+M-CLIP, 한글=M-CLIP 전담."""
        fused = {}
        if not ko_fashion.has_korean(q):
            for rank, idx in enumerate(clip_rank(q, 200)):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(mclip_rank(q, 200)):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        return [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:TOPK]]

    agg = {k: {"base": [], "fused": []} for k in ("EN", "KO")}
    rows = []
    for en, ko, color, items in QUERY_SET:
        be = precision_at_k(clip_rank(en, TOPK), names, color, items)
        fe = precision_at_k(fused_rank(en),      names, color, items)
        bk = precision_at_k(clip_rank(ko, TOPK), names, color, items)
        fk = precision_at_k(fused_rank(ko),      names, color, items)
        agg["EN"]["base"].append(be); agg["EN"]["fused"].append(fe)
        agg["KO"]["base"].append(bk); agg["KO"]["fused"].append(fk)
        rows.append((en, ko, be, fe, bk, fk))

    print("\n  쿼리                         | EN P@10 base→fused | KO P@10 base→fused")
    print("  " + "-" * 70)
    for en, ko, be, fe, bk, fk in rows:
        print(f"  {en:18s}/{ko:9s} |   {be:.2f} → {fe:.2f}       |   {bk:.2f} → {fk:.2f}")

    summary = {lang: {"baseline_p@10": round(float(np.mean(agg[lang]["base"])), 4),
                      "fused_p@10":    round(float(np.mean(agg[lang]["fused"])), 4)}
               for lang in ("EN", "KO")}

    print("\n  ===== 평균 Precision@10 (baseline CLIP text.index → 배포 fused) =====")
    for lang in ("EN", "KO"):
        s = summary[lang]
        print(f"  [{lang}] baseline {s['baseline_p@10']:.3f}  →  fused {s['fused_p@10']:.3f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"topk": TOPK, "n_queries": len(QUERY_SET), "summary": summary,
                   "paths": "baseline=CLIP text.index / fused=EN:CLIP+M-CLIP, KO:M-CLIP",
                   "relevance": "상품명에 색상 AND 품목동의어 포함 시 relevant (메타데이터 자동 라벨)"},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
