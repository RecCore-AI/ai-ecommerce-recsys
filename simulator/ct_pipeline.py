import os
import json
import time
import datetime
import shutil
import subprocess
import requests
# ---------------------------------------------------------
# ⚙️ CT 파이프라인 환경 설정
# ---------------------------------------------------------
NEW_LOGS_PATH = "data/new_train_logs.csv"  # 실시간 시뮬레이터가 쏘는 로그
ARCHIVE_DIR = "data/archive/"              # 재학습이 끝난 로그를 보관할 창고
MODELS_DIR = "data/models/"                # 모델 가중치 폴더
METRICS_PATH = "data/metrics.json"         # 오프라인 평가 지표 (성능 모니터링용)
THRESHOLD = 10000                          # 재학습 트리거 임계값 (명세서 기준 1만건)
PERF_THRESHOLDS = {"hitrate_50": 0.20, "auc": 0.70}  # 추천 성능 임계값 (명세 목표치)

# 필요한 폴더가 없으면 자동 생성
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def count_new_logs():
    """현재 쌓인 신규 로그 파일의 줄 수를 셉니다."""
    if not os.path.exists(NEW_LOGS_PATH):
        return 0
    with open(NEW_LOGS_PATH, 'r', encoding='utf-8') as f:
        # 첫 줄은 헤더이므로 제외하고 카운트
        count = sum(1 for _ in f) - 1 
    return max(0, count)

def trigger_retraining():
    """로그가 1만 건을 넘으면 독립된 프로세스로 재학습을 가동합니다."""
    print(f"\n🚨 [CT Trigger] 신규 로그 {THRESHOLD}건 축적 돌파!")
    print("🧠 별도의 워커 프로세스를 띄워 DeepFM 미세조정(Fine-tuning)을 가동합니다...")
    
    # 💡 [핵심: MLOps 정석] 모니터링 데몬이 죽지 않도록, 실제 학습은 별도 파일로 실행
    try:
        subprocess.run(["python", "phase4_retrain_job.py"], check=True)
        print("✅ DeepFM 재학습 프로세스 정상 종료!")

        # 💡 [여기 추가] 학습 끝나자마자 API 서버 찔러서 뇌 교체하기!
        try:
            response = requests.post("http://api-server:8000/api/reload-model")
            print(f"🔄 API 서버 뇌 교체 완료: {response.json().get('message', '성공')}")
        except Exception as e:
            print(f"⚠️ API 갱신 실패 (서버가 켜져있는지 확인하세요): {e}")

    except subprocess.CalledProcessError:
        print("❌ [에러] 재학습 워커 프로세스에서 문제가 발생했습니다. 파이프라인을 중단합니다.")
        return

    # ---------------------------------------------------------
    # 📦 데이터 아카이빙 및 버전 관리
    # ---------------------------------------------------------
    # 어떤 시점의 데이터로 학습했는지 타임스탬프 기록
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file = os.path.join(ARCHIVE_DIR, f"logs_used_for_v{version}.csv")
    
    # 방금 학습에 쓴 1만 건의 로그는 창고로 이동 (다음 모니터링 때 중복 카운트 방지)
    shutil.move(NEW_LOGS_PATH, archive_file)
    print(f"🧹 처리된 로그 데이터는 안전하게 백업되었습니다. (파일명: {os.path.basename(archive_file)})\n")
    print(f"👁️ [CT Monitor] 다시 새로운 실시간 트래픽을 감시합니다...")

def check_performance():
    """metrics.json 의 추천 성능 지표를 점검하고, 임계값 이하로 떨어지면 알림을 출력한다.
    (명세: 추천 성능 지표(HitRate/CTR) 모니터링 + 임계값 이하 시 성능 저하 알림)"""
    if not os.path.exists(METRICS_PATH):
        return
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            rec = json.load(f).get("recommend", {})
    except Exception:
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts = []
    for metric, thr in PERF_THRESHOLDS.items():
        val = rec.get(metric)
        if val is not None and val < thr:
            alerts.append(f"{metric} = {val:.4f}  (임계값 {thr} 이하)")

    if alerts:
        print(f"\n🔔 [CT Monitor] {now}")
        for a in alerts:
            print(f"   현재 {a}")
        print("   ⚠️ 성능 저하 감지! 재학습을 권장합니다.\n")
    else:
        hr, auc = rec.get("hitrate_50"), rec.get("auc")
        print(f"📈 [CT Monitor] {now}  추천 성능 정상 (HitRate@50={hr}, AUC={auc})")


def run_monitor():
    """무한 루프를 돌며 ① 추천 성능 지표 ② 신규 로그 누적량을 감시하는 데몬 함수"""
    print("👁️ [CT Monitor] 실시간 데이터 모니터링 데몬 시작...")

    tick = 0
    while True:
        # ① 성능 지표 모니터링 — 약 1분마다 (metrics.json 은 자주 바뀌지 않음)
        if tick % 12 == 0:
            check_performance()

        # ② 신규 로그 누적량 — 1만 건을 넘기면 재학습 트리거
        log_count = count_new_logs()
        if log_count >= THRESHOLD:
            trigger_retraining()
        else:
            print(f"📊 [CT Monitor] 현재 신규 로그: {log_count}건 / 임계값: {THRESHOLD}건 (대기 중...)")

        tick += 1
        # 서버 자원을 덜 쓰기 위해 5초마다 상태 체크
        time.sleep(5)

if __name__ == "__main__":
    run_monitor()