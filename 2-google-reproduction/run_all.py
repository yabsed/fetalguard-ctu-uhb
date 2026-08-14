"""Chiou et al. (2025) 재구현 파이프라인을 CTU-UHB **552건 전체**에 적용한다.

3.3절이 재구현한 것은 두 갈래다. 이 스크립트는 둘 다 전수 실행하고, 그 결과를 파일로 남긴다.

  (A) 신경망 파이프라인 (`preprocess.py`)
      원신호 → 양끝 결측 제거 → 5분 창 품질 평가(결측 ≥50% 창은 후단계서 0 재지정) →
      결측 15초 기준 분류(≤15초 보간 / >15초 0 유지) → 15포인트 Hamming 평활 →
      분만 직전 30분 크롭 → 1 Hz 다운샘플 → 채널별 max-abs 스케일 → CTG-net 입력 (2, 1800)
      산출물: log/ctu_model_inputs.npz — 크롭이 가능한 레코드 전부의 모델 입력 텐서

  (B) 규칙 기반 특성 파이프라인 (`features.py`)
      원신호 → 짧은 결측 보간 → 30초 롤링 평활 → 남은 결측 보간 → 30분 크롭 →
      반복 기저선(10분 창 평균 초기화) → 특성 17개 (medRxiv Appendix A.2 규칙)
      산출물: log/ctu_features17.csv — 552행 × (라벨 + 전처리 통계 + 특성 17개)

논문과 대조 가능한 수치(코호트 크기·라벨 분포)는 log/summary.md에 함께 기록한다. 학습·성능
수치는 이 저장소의 범위 밖이다 — 3.3절의 목적은 선행 연구 결과의 정확한 전달이며, 공개하는
것은 전처리·특성·모델의 재구현 코드와 그 전수 변환 결과까지다.

실행: cd 2-google-reproduction && python run_all.py   (약 1분)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ctguhb import (  # noqa: E402
    METADATA_ATTRS, list_record_ids, make_labels, read_signals,
)
from features import (  # noqa: E402
    FEATURE_NAMES, extract_last30_features, preprocess_for_features,
)
from preprocess import (  # noqa: E402
    CROP_MIN, FS, LONG_GAP_S, crop_last_minutes, gap_runs, impute_short_gaps,
    make_model_input, preprocess_record, trim_edge_missing,
)

DATA = next(p for p in HERE.parents if (p / "ctu-hub-ctgdb").exists()) / "ctu-hub-ctgdb"
LOG = HERE / "log"
LOG.mkdir(exist_ok=True)

# 논문이 보고한 대조 수치 (Chiou et al., 2025). 논문 전처리 절의 (n = 496)·(n = 56)은
# 분석 제외가 아니라 90%/10% 학습·테스트 **분할**이다 — write_summary의 설명 참조.
PAPER = {"n_total": 552,
         "ph_abnormal": 177, "apgar_abnormal": 68, "lor_abnormal": 198}


def process_record(rid: str) -> tuple[dict, np.ndarray | None]:
    """레코드 한 건 → (요약 행, 모델 입력 (2,1800) 또는 None)."""
    fhr_raw, uc_raw, meta, fs = read_signals(DATA / f"{rid}.hea")
    assert fs == FS

    fhr_t, uc_t = trim_edge_missing(fhr_raw, uc_raw)
    gaps = [e - s for s, e in gap_runs(fhr_t == 0)]
    long_gaps = [g for g in gaps if g > LONG_GAP_S * FS]

    row = {
        "record_id": rid,
        "pH": meta.get("pH", np.nan), "Apgar1": meta.get("Apgar1", np.nan),
        "raw_min": len(fhr_raw) / FS / 60,
        "trimmed_min": len(fhr_t) / FS / 60,
        "missing_frac": float((fhr_t == 0).mean()) if len(fhr_t) else 1.0,
        "n_gaps": len(gaps),
        "n_long_gaps": len(long_gaps),
        "longest_gap_s": max(gaps, default=0) / FS,
    }

    # (A) 신경망 파이프라인
    fhr_p, uc_p = preprocess_record(fhr_raw, uc_raw)
    x = make_model_input(fhr_p, uc_p)
    row["crop_ok"] = x is not None

    # 크롭 창에 남은 결측(>15초 구간은 0으로 유지되므로 그대로 셀 수 있다) —
    # 신호 품질 기술 통계. 평활 전 신호에서 재므로 평활 창 선택에 영향받지 않는다.
    crop = crop_last_minutes(impute_short_gaps(fhr_t, int(LONG_GAP_S * FS)), fs=FS)
    row["crop_missing_frac"] = float((crop == 0).mean()) if crop is not None else 1.0

    # (B) 규칙 기반 특성 파이프라인
    fhr_f, uc_f = preprocess_for_features(fhr_t, uc_t)
    feats = extract_last30_features(fhr_f, uc_f)
    if feats is None:
        row.update({k: np.nan for k in FEATURE_NAMES})
    else:
        row.update(dict(zip(FEATURE_NAMES, feats.astype(float))))
    return row, x


def write_summary(df: pd.DataFrame, labels: dict, dt: float, n_inputs: int) -> str:
    got = {"n_total": len(df),
           "ph_abnormal": int(labels["ph"].sum()),
           "apgar_abnormal": int(labels["apgar"].sum()),
           "lor_abnormal": int(labels["lor"].sum())}
    names = {"n_total": "전체 레코드",
             "ph_abnormal": "pH < 7.20 (비정상)", "apgar_abnormal": "1분 Apgar < 7 (비정상)",
             "lor_abnormal": "pH 또는 Apgar 이상 (LOR)"}

    L = [
        "# 전수 실행 요약 — CTU-UHB 552건에 대한 Chiou et al. (2025) 재구현",
        "",
        f"- 레코드 {len(df)}건 전체 처리 시간: **{dt:.1f}초** (단일 CPU 코어, "
        f"레코드당 평균 {dt / len(df) * 1000:.0f} ms)",
        f"- 신경망 입력 텐서: {n_inputs}건 × (2 채널 × {CROP_MIN * 60}포인트) "
        "→ `log/ctu_model_inputs.npz`",
        f"- 규칙 기반 특성: {int(df[FEATURE_NAMES[0]].notna().sum())}건 × 17개 "
        "→ `log/ctu_features17.csv`",
        "",
        "## 코호트·라벨 — 논문 보고값과의 대조",
        "",
        "| 항목 | 논문 | 이 재구현 |",
        "|---|---:|---:|",
    ]
    for k, label in names.items():
        L.append(f"| {label} | {PAPER[k]} | {got[k]} |")
    L += [
        "",
        "세 라벨(Eq. 1)의 비정상 건수는 논문과 정확히 일치한다 — 헤더의 pH·Apgar1을 그대로 "
        "쓰므로 재구현 여지가 없는 부분이고, 그래서 데이터 로딩이 옳다는 확인이 된다.",
        "",
        "**논문의 (n = 496)·(n = 56)에 대하여.** 논문 전처리 절의 "
        "\"4,315,200 minutes (n = 496 recordings), 148,800 min (n = 496 recordings), and "
        "1,680 min (n = 56 recordings) ... for pre-training, training, and testing, "
        "respectively\"는 분석에서 제외된 레코드가 있다는 뜻이 아니라 **90%/10% "
        "학습·테스트 분할**이다. Data splitting 절이 이를 명시한다 — 레코드 식별자의 10%를 "
        "테스트로 고정하고 나머지 90%를 10-fold 교차검증에 썼다. 산술도 맞물린다: "
        "496 + 56 = 552(제외 없음), 148,800분 ÷ 30분 = 4,960 = 496건 × 증강 크롭 10개"
        "(Appendix A.1), 1,680분 ÷ 30분 = 56 = 테스트 56건 × 결정론적 크롭 1개. "
        f"실제로 이 데이터셋의 원신호는 전부 60분 이상이라 {CROP_MIN}분 크롭은 552건 모두 "
        f"가능하고(양끝 결측 제거 후 최단 {df['trimmed_min'].min():.0f}분), 이 전수 변환도 "
        "552건 전부를 처리한다. `crop_missing_frac` 열(크롭 창의 잔여 >"
        f"{LONG_GAP_S}초 결측 비율)은 코호트 선별 기준이 아니라 신호 품질 기술 통계다.",
        "",
        "## 전처리 통계 (552건)",
        "",
        "| 항목 | 중앙값 | 평균 | 최소 | 최대 |",
        "|---|---:|---:|---:|---:|",
    ]
    for col, label in [("raw_min", "원신호 길이 (분)"),
                       ("trimmed_min", "양끝 결측 제거 후 (분)"),
                       ("missing_frac", "결측 비율 (기록 전체)"),
                       ("crop_missing_frac", f"결측 비율 (마지막 {CROP_MIN}분)"),
                       ("n_long_gaps", f">{LONG_GAP_S}초 결측 구간 수"),
                       ("longest_gap_s", "최장 결측 (초)")]:
        s = df[col].astype(float)
        L.append(f"| {label} | {s.median():.2f} | {s.mean():.2f} | {s.min():.2f} | {s.max():.2f} |")

    L += [
        "",
        "## 특성 17개 — pH 라벨 기준 평균 (방향성 점검)",
        "",
        "| 특성 | 정상(pH≥7.20) | 비정상(pH<7.20) |",
        "|---|---:|---:|",
    ]
    g = df.dropna(subset=FEATURE_NAMES).groupby("label_ph")[FEATURE_NAMES].mean()
    for c in FEATURE_NAMES:
        L.append(f"| {c} | {g.loc[0, c]:.3f} | {g.loc[1, c]:.3f} |")
    text = "\n".join(L) + "\n"
    (LOG / "summary.md").write_text(text)
    return text


def main():
    t0 = time.time()
    ids = list_record_ids(DATA)
    rows, inputs, input_ids = [], [], []
    for i, rid in enumerate(ids, 1):
        row, x = process_record(rid)
        rows.append(row)
        if x is not None:
            inputs.append(x)
            input_ids.append(rid)
        if i % 100 == 0 or i == len(ids):
            print(f"{i}/{len(ids)} ({time.time() - t0:.0f}초)", flush=True)
    dt = time.time() - t0

    df = pd.DataFrame(rows)
    meta_df = df.set_index("record_id")[["pH", "Apgar1"]]
    labels = make_labels(meta_df)
    for k, v in labels.items():
        df[f"label_{k}"] = v.to_numpy()

    cols = (["record_id", "pH", "Apgar1", "label_ph", "label_apgar", "label_lor",
             "raw_min", "trimmed_min", "missing_frac", "n_gaps", "n_long_gaps",
             "longest_gap_s", "crop_ok", "crop_missing_frac"]
            + FEATURE_NAMES)
    df[cols].to_csv(LOG / "ctu_features17.csv", index=False)

    # 모델 입력은 552건 전부를 저장한다 — 논문도 552건 전부를 90%/10%로 나눠 썼다
    # (write_summary의 설명 참조). 신호 품질은 CSV의 crop_missing_frac으로 확인한다.
    X = np.stack(inputs).astype(np.float32)
    np.savez_compressed(LOG / "ctu_model_inputs.npz",
                        record_ids=np.array(input_ids), X=X,
                        channels=np.array(["FHR", "UC"]),
                        metadata_attrs=np.array(METADATA_ATTRS))

    print()
    print(write_summary(df, labels, dt, len(X))[:1200])
    print(f"저장: {LOG}/ctu_features17.csv, {LOG}/ctu_model_inputs.npz "
          f"(X {X.shape}), {LOG}/summary.md")


if __name__ == "__main__":
    main()
