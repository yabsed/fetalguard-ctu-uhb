"""CTU-UHB Intrapartum CTG Database 리더 (wfdb 패키지 불필요).

이 DB의 552개 레코드는 전부 WFDB **format 16**(16-bit 부호있는 정수,
little-endian, 채널 다중화)으로 저장돼 있어 numpy만으로 읽을 수 있다.
파서·체크섬 검증 로직은 ../1-dataset-walkthrough/ctu_uhb_walkthrough.ipynb 에서
552건 전체에 대해 검증했다.

데이터: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
원논문: Chudáček et al., BMC Pregnancy and Childbirth 14:16 (2014).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

FS = 4  # Hz — 전 레코드 공통 (헤더 1행의 세 번째 필드)

# 논문(Chiou et al., 2025)의 메타데이터 입력 벡터 11개 속성 (Fig. 2g 기준)
METADATA_ATTRS = [
    "Gest. weeks", "Age", "Gravidity", "Parity", "Diabetes", "Hypertension",
    "Preeclampsia", "Liq. praecox", "Pyrexia", "Meconium", "Induced",
]


def parse_header(hea_path):
    """WFDB .hea → (record dict, signals list, meta dict).

    record:  {"name", "n_signals", "fs", "n_samples"}
    signals: [{"file","format","gain","unit","adc_zero","init_value","checksum","description"}]
    meta:    헤더 주석(#)의 임상 변수 {키: float}. 절 제목(`-- ...`)과 빈 줄은 건너뛴다.
    """
    hea_path = Path(hea_path)
    lines = hea_path.read_text().splitlines()
    rec = lines[0].split()
    record = {"name": rec[0], "n_signals": int(rec[1]),
              "fs": float(rec[2]), "n_samples": int(rec[3])}

    signals, meta = [], {}
    for line in lines[1:]:
        if not line.strip():
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if text.startswith("-"):
                continue
            m = re.match(r"^(.*?)\s+(-?[\d.]+|NaN)$", text)
            if m:
                meta[m.group(1).strip()] = float(m.group(2))
            continue
        p = line.split()
        gain = re.match(r"^([\d.]+)", p[2])
        signals.append({
            "file": p[0], "format": int(p[1]),
            "gain": float(gain.group(1)), "unit": p[2].partition("/")[2],
            "adc_zero": int(p[4]), "init_value": int(p[5]),
            "checksum": int(p[6]), "description": p[8],
        })
    return record, signals, meta


def read_signals(hea_path):
    """한 레코드를 읽어 (fhr, uc, meta, fs) 반환. 체크섬·첫 샘플로 읽기 검증."""
    record, signals, meta = parse_header(hea_path)
    assert record["n_signals"] == 2 and record["fs"] == FS
    raw = np.fromfile(Path(hea_path).with_suffix(".dat"), dtype="<i2")
    raw = raw.reshape(-1, record["n_signals"])

    values = {}
    for i, s in enumerate(signals):
        assert s["format"] == 16, f"예상 밖 포맷: {s}"
        digital = raw[:, i].astype(np.int64)
        assert digital[0] == s["init_value"], f"{s['description']} 첫 샘플 불일치"
        assert digital.sum() % 65536 == s["checksum"] % 65536, f"{s['description']} 체크섬 불일치"
        assert s["adc_zero"] == 0
        values[s["description"]] = digital / s["gain"]
    return values["FHR"], values["UC"], meta, record["fs"]


def list_record_ids(data_dir):
    return (Path(data_dir) / "RECORDS").read_text().split()


def load_all(data_dir):
    """552건 전체를 읽어 {record_id: dict} 반환. 수 초 소요, 메모리 ~200MB."""
    data_dir = Path(data_dir)
    out = {}
    for rid in list_record_ids(data_dir):
        fhr, uc, meta, fs = read_signals(data_dir / f"{rid}.hea")
        out[rid] = {"fhr": fhr, "uc": uc, "meta": meta, "fs": fs}
    return out


def metadata_frame(records):
    """레코드 dict → 메타데이터 DataFrame (행=레코드)."""
    rows = {rid: r["meta"] for rid, r in records.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def make_labels(meta_df):
    """Chiou et al. (2025) Eq. (1)의 세 가지 라벨.

    ph    : 제대동맥혈 pH < 7.20        → 비정상 177 / 정상 375
    apgar : 1분 Apgar < 7               → 비정상  68 / 정상 484
    lor   : ph 이상 OR apgar 이상       → 비정상 198 / 정상 354
    반환: {"ph": Series(int), "apgar": Series, "lor": Series} (인덱스=레코드 ID)
    """
    ph = (meta_df["pH"] < 7.20).astype(int)
    apgar = (meta_df["Apgar1"] < 7).astype(int)
    return {"ph": ph, "apgar": apgar, "lor": (ph | apgar).astype(int)}


def metadata_vector(meta_df, rid):
    """논문의 메타데이터 입력 벡터 (11속성). 결측은 0으로 두고 스케일링은 학습 시 수행."""
    row = meta_df.loc[rid, METADATA_ATTRS].to_numpy(dtype=np.float64)
    return np.nan_to_num(row, nan=0.0).astype(np.float32)
