"""
한↔영 패션 도메인 사전 (검색 한국어 지원 + Query Understanding 공용)

왜 범용 MT 대신 사전인가:
  실측에서 m2m100 같은 범용 번역기는 패션 명사를 망가뜨린다
  (원피스→"One Piece", 청바지→"Jewelry", 운동화→"exercise").
  검색 도메인은 [색][품목][소재/핏] 조합이 대부분이라, 도메인 사전으로
  핵심 토큰만 결정적으로 치환하면 명사 환각 없이 한국어 의미검색이 성립한다.

제공:
  - normalize_query(q): 한글 쿼리의 알려진 토큰을 영어로 치환 (M-CLIP 입력용)
  - extract_filters(q): 가격/색상/카테고리 필터 추출 (Query Understanding, 4번에서 사용)
  - COLOR / CATEGORY 사전: 결과 후처리 필터에서 재사용
"""
import re

# ── 색상 (검색 + 필터 공용) ────────────────────────────────
COLOR = {
    "검정": "black", "검은": "black", "블랙": "black",
    "흰": "white", "흰색": "white", "하양": "white", "화이트": "white",
    "빨강": "red", "빨간": "red", "레드": "red",
    "파랑": "blue", "파란": "blue", "블루": "blue", "남색": "navy", "네이비": "navy",
    "초록": "green", "초록색": "green", "그린": "green", "카키": "khaki",
    "노랑": "yellow", "노란": "yellow", "옐로": "yellow",
    "분홍": "pink", "핑크": "pink",
    "보라": "purple", "퍼플": "purple",
    "회색": "grey", "그레이": "grey", "회": "grey",
    "갈색": "brown", "브라운": "brown", "베이지": "beige",
    "주황": "orange", "오렌지": "orange",
}

# ── 품목/카테고리 (검색용 영어 명사) ──────────────────────
CATEGORY = {
    "원피스": "dress", "드레스": "dress",
    "후드티": "hoodie", "후드": "hoodie",
    "맨투맨": "sweatshirt", "스웨트셔츠": "sweatshirt",
    "티셔츠": "t-shirt", "티": "t-shirt", "반팔티": "t-shirt",
    "셔츠": "shirt", "블라우스": "blouse",
    "청바지": "jeans", "진": "jeans", "데님": "denim",
    "바지": "trousers", "팬츠": "trousers", "슬랙스": "trousers",
    "반바지": "shorts", "레깅스": "leggings",
    "치마": "skirt", "스커트": "skirt",
    "니트": "knit", "스웨터": "sweater", "점퍼": "jumper", "가디건": "cardigan", "카디건": "cardigan",
    "자켓": "jacket", "재킷": "jacket", "코트": "coat", "패딩": "padded jacket", "조끼": "vest",
    "가방": "bag", "백": "bag", "토트백": "tote bag", "숄더백": "shoulder bag", "백팩": "backpack",
    "운동화": "sneakers", "스니커즈": "sneakers", "신발": "shoes", "구두": "shoes", "부츠": "boots",
    "양말": "socks", "모자": "hat", "캡": "cap", "벨트": "belt", "스카프": "scarf", "장갑": "gloves",
    "수영복": "swimwear", "비키니": "bikini", "속옷": "underwear", "잠옷": "pajamas", "브라": "bra",
}

# ── 소재/핏/속성 (검색 보조) ──────────────────────────────
ATTRIBUTE = {
    "가죽": "leather", "면": "cotton", "울": "wool", "실크": "silk", "린넨": "linen", "캐시미어": "cashmere",
    "오버핏": "oversized", "오버사이즈": "oversized", "루즈핏": "loose", "슬림": "slim", "스키니": "skinny",
    "긴팔": "long sleeve", "반팔": "short sleeve", "민소매": "sleeveless",
    "하이웨스트": "high waist", "크롭": "cropped",
}

_LEXICON = {**COLOR, **CATEGORY, **ATTRIBUTE}
# 긴 단어 우선 치환(부분일치 충돌 방지)
_TERMS = sorted(_LEXICON.keys(), key=len, reverse=True)
_HANGUL = re.compile(r"[가-힣]")

# 가격: "10만원 이하", "5만원 이하", "30000원 미만" 등
_PRICE_MAX = re.compile(r"(\d+)\s*만\s*원?\s*(이하|미만|아래|밑)")
_PRICE_MAX_WON = re.compile(r"(\d{4,})\s*원?\s*(이하|미만|아래|밑)")
_PRICE_MIN = re.compile(r"(\d+)\s*만\s*원?\s*(이상|초과|위)")


def has_korean(text: str) -> bool:
    return bool(_HANGUL.search(str(text)))


def normalize_query(query: str) -> str:
    """한글 쿼리의 알려진 패션 토큰을 영어로 치환해 M-CLIP 입력 문자열을 만든다.
    영어 쿼리는 그대로 통과. 모르는 한글 토큰도 그대로 둔다(부분 신호 보존)."""
    q = str(query)
    if not has_korean(q):
        return q
    out = q
    for term in _TERMS:
        if term in out:
            out = out.replace(term, " " + _LEXICON[term] + " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def strip_price(query: str) -> str:
    """가격 표현(이미 필터로 추출됨)을 검색 쿼리에서 제거한다.
    숫자가 BM25 어휘검색을 오염시키는 것 방지: 예) '5만원 이하 바지'의 '5'가
    상품명 속 '5'(주얼리·벨트 등)에 매칭돼 상위로 올라오는 문제."""
    q = str(query)
    for rx in (_PRICE_MAX, _PRICE_MAX_WON, _PRICE_MIN):
        q = rx.sub(" ", q)
    return re.sub(r"\s+", " ", q).strip()


def extract_filters(query: str) -> dict:
    """자연어 쿼리에서 구조화 필터 추출 (Query Understanding, 4번).
    예: '10만원 이하 검정 니트' -> {price_max:100000, color:'black', category:'knit'}"""
    q = str(query)
    f = {}

    m = _PRICE_MAX.search(q)
    if m:
        f["price_max"] = int(m.group(1)) * 10000
    else:
        m = _PRICE_MAX_WON.search(q)
        if m:
            f["price_max"] = int(m.group(1))
    m = _PRICE_MIN.search(q)
    if m:
        f["price_min"] = int(m.group(1)) * 10000

    for ko, en in COLOR.items():
        if ko in q:
            f["color"] = en
            break
    for ko, en in CATEGORY.items():
        if ko in q:
            f.setdefault("category", en)
            break
    return f
