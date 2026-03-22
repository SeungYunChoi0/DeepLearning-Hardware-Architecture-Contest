#!/usr/bin/env python3
"""
AIX2026 Quantization Multiplier Auto-Tuner
전략: Greedy Hill-Climbing (레이어별 순차 최적화)
      → 최대 22×6 = 132회 실행으로 로컬 최적 탐색
"""

import os
import re
import subprocess
import copy
import json
import time
from datetime import datetime
from itertools import product

# ============================================================
# 경로 설정 (Mac 환경 기준)
# ============================================================
SKELETON_DIR = "/Volumes/Yun_ssd/AIX2026/skeleton"
BIN_DIR      = os.path.join(SKELETON_DIR, "bin")
C_SRC_FILE   = "/Volumes/Yun_ssd/AIX2026/skeleton/src/yolov2_forward_network_quantized.c"
TEST_SCRIPT  = "sh script-unix-aix2024-test-all-quantized.sh"
LOG_FILE     = "./aix_tuner_log.json"
BEST_FILE    = "./aix_best_result.json"

# ============================================================
# 탐색 공간 설정
# ============================================================
VALID_VALUES  = [16, 32, 64, 128, 256, 512]   # 2의 지수승, 16~512
BASELINE_MAP  = 79.28                           # 현재 베이스라인 mAP
TARGET_MAP    = BASELINE_MAP                    # 이것보다 높아야 저장

# 현재 베이스라인 multiplier 값 (코드에서 확인된 값)
BASELINE_WEIGHT = [32,  256, 256, 256, 256, 256, 256, 256, 256, 256, 512]
BASELINE_INPUT  = [128,  32,  32,  32,  32,  16,  32,  32,  64,  64,  32]

TOTAL_LAYERS = 11
LAYER_NAMES  = [f"conv{i}" for i in [0,2,4,6,8,10,12,13,14,17,20]]

# ============================================================
# C 소스 파일 파라미터 수정
# ============================================================
def update_c_source(weight_mults, input_mults):
    """C 소스의 multiplier 배열을 새 값으로 교체"""
    with open(C_SRC_FILE, 'r') as f:
        content = f.read()

    def build_array_str(name, values, indent="\t", comment_layers=None):
        """배열 선언 문자열 생성"""
        if comment_layers is None:
            comment_layers = LAYER_NAMES
        lines = []
        for i, (v, lname) in enumerate(zip(values, comment_layers)):
            comma = "," if i < len(values) - 1 else ""
            lines.append(f"{indent}{v}{comma}\t  //conv {lname.replace('conv','')}")
        return "\n".join(lines)

    # weight_quant_multiplier 배열 교체
    w_vals = ",\n\t".join(
        [f"{v}\t  //{LAYER_NAMES[i]}" for i, v in enumerate(weight_mults)]
    )
    # input_quant_multiplier 배열 교체
    i_vals = ",\n\t".join(
        [f"{v}\t  //{LAYER_NAMES[i]}" for i, v in enumerate(input_mults)]
    )

    # 정규식으로 배열 내용 교체
    # weight_quant_multiplier 배열
    w_pattern = r'(float weight_quant_multiplier\[TOTAL_CALIB_LAYER\]\s*=\s*\{)[^}]*(};)'
    w_body = "\n\t" + ",\n\t".join(
        [f"{v}\t  //conv {LAYER_NAMES[i].replace('conv','')}" for i, v in enumerate(weight_mults)]
    ) + "\n\t"
    content = re.sub(w_pattern, r'\g<1>' + w_body + r'\2', content, flags=re.DOTALL)

    # input_quant_multiplier 배열
    i_pattern = r'(float input_quant_multiplier\[TOTAL_CALIB_LAYER\]\s*=\s*\{)[^}]*(};)'
    i_body = "\n\t " + ",\n\t ".join(
        [f"{v}\t  //conv {LAYER_NAMES[i].replace('conv','')}" for i, v in enumerate(input_mults)]
    ) + "\n\t"
    content = re.sub(i_pattern, r'\g<1>' + i_body + r'\2', content, flags=re.DOTALL)

    with open(C_SRC_FILE, 'w') as f:
        f.write(content)

def verify_c_source(weight_mults, input_mults):
    """교체된 값이 실제로 C 파일에 들어갔는지 확인"""
    with open(C_SRC_FILE, 'r') as f:
        content = f.read()
    for v in weight_mults:
        if str(v) not in content:
            return False
    return True

# ============================================================
# 빌드 & 실행
# ============================================================
def build_project():
    """make clean && make 실행"""
    print("  [Build] make clean && make ...", end=" ", flush=True)
    result = subprocess.run(
        "make clean && make",
        shell=True, cwd=SKELETON_DIR,
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr[-500:])
        return False
    print("OK")
    return True

def run_test():
    """테스트 스크립트 실행 후 mAP 파싱"""
    print("  [Test ] Running...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run(
        TEST_SCRIPT,
        shell=True, cwd=BIN_DIR,
        capture_output=True, text=True, timeout=300
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr

    # mAP 파싱: "mean average precision (mAP) = 0.792752, or 79.28 %"
    match = re.search(r'mean average precision \(mAP\)\s*=\s*([\d.]+),\s*or\s*([\d.]+)\s*%', output)
    if match:
        map_val = float(match.group(2))
        print(f"mAP = {map_val:.4f}%  ({elapsed:.0f}s)")
        return map_val, output
    else:
        print(f"PARSE FAILED ({elapsed:.0f}s)")
        print("  Last 300 chars:", output[-300:])
        return None, output

# ============================================================
# 로그 저장
# ============================================================
def save_log(log_entries):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_entries, f, indent=2)

def save_best(best_map, weight_mults, input_mults):
    with open(BEST_FILE, 'w') as f:
        json.dump({
            "map": best_map,
            "weight_quant_multiplier": weight_mults,
            "input_quant_multiplier": input_mults,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"\n  ★ 새 최고 기록 저장: mAP = {best_map:.4f}%  →  {BEST_FILE}")

# ============================================================
# 전략 1: Greedy Hill-Climbing (권장 - 기본 실행)
# ============================================================
def greedy_search(start_weight=None, start_input=None):
    """
    각 레이어의 multiplier를 하나씩 순차적으로 최적화.
    한 pass에서 개선이 없으면 종료.
    """
    w_cur = list(start_weight or BASELINE_WEIGHT)
    i_cur = list(start_input  or BASELINE_INPUT)
    best_map = BASELINE_MAP
    log_entries = []
    trial_no = 0

    print("\n" + "="*60)
    print("  Greedy Hill-Climbing 시작")
    print(f"  베이스라인 mAP: {best_map:.4f}%")
    print("="*60)

    for pass_no in range(1, 4):   # 최대 3 pass
        print(f"\n  ── Pass {pass_no} ──")
        improved_this_pass = False

        # 1) weight multiplier 최적화
        for layer_idx in range(TOTAL_LAYERS):
            layer_name = LAYER_NAMES[layer_idx]
            best_w_val = w_cur[layer_idx]

            for val in VALID_VALUES:
                if val == w_cur[layer_idx]:
                    continue  # 현재 값은 건너뜀

                trial_no += 1
                w_trial = list(w_cur)
                w_trial[layer_idx] = val

                print(f"\n  [{trial_no:03d}] weight[{layer_name}] = {w_cur[layer_idx]} → {val}")
                update_c_source(w_trial, i_cur)
                if not build_project():
                    continue
                map_val, raw_output = run_test()
                if map_val is None:
                    continue

                entry = {
                    "trial": trial_no, "type": "weight", "layer": layer_name,
                    "value": val, "map": map_val,
                    "weight": list(w_trial), "input": list(i_cur),
                    "timestamp": datetime.now().isoformat()
                }
                log_entries.append(entry)
                save_log(log_entries)

                if map_val > best_map:
                    best_map = map_val
                    best_w_val = val
                    w_cur = list(w_trial)
                    save_best(best_map, w_cur, i_cur)
                    improved_this_pass = True

            # 레이어 최적 후 현재값으로 복원 확인
            if w_cur[layer_idx] != best_w_val:
                w_cur[layer_idx] = best_w_val

        # 2) input multiplier 최적화
        for layer_idx in range(TOTAL_LAYERS):
            layer_name = LAYER_NAMES[layer_idx]
            best_i_val = i_cur[layer_idx]

            for val in VALID_VALUES:
                if val == i_cur[layer_idx]:
                    continue

                trial_no += 1
                i_trial = list(i_cur)
                i_trial[layer_idx] = val

                print(f"\n  [{trial_no:03d}] input[{layer_name}] = {i_cur[layer_idx]} → {val}")
                update_c_source(w_cur, i_trial)
                if not build_project():
                    continue
                map_val, raw_output = run_test()
                if map_val is None:
                    continue

                entry = {
                    "trial": trial_no, "type": "input", "layer": layer_name,
                    "value": val, "map": map_val,
                    "weight": list(w_cur), "input": list(i_trial),
                    "timestamp": datetime.now().isoformat()
                }
                log_entries.append(entry)
                save_log(log_entries)

                if map_val > best_map:
                    best_map = map_val
                    best_i_val = val
                    i_cur = list(i_trial)
                    save_best(best_map, w_cur, i_cur)
                    improved_this_pass = True

            if i_cur[layer_idx] != best_i_val:
                i_cur[layer_idx] = best_i_val

        if not improved_this_pass:
            print(f"\n  Pass {pass_no}: 개선 없음 → Greedy 탐색 종료")
            break

    # 최종 복원
    print("\n  최종 최적 설정으로 C 소스 복원 중...")
    update_c_source(w_cur, i_cur)
    build_project()

    return best_map, w_cur, i_cur, log_entries

# ============================================================
# 전략 2: Random Search (Greedy 후 추가 탐색용)
# ============================================================
def random_search(best_weight, best_input, best_map, n_trials=200):
    """랜덤하게 multiplier 조합 시도"""
    import random
    log_entries = []
    trial_no = 0

    print("\n" + "="*60)
    print(f"  Random Search 시작 ({n_trials}회)")
    print("="*60)

    for _ in range(n_trials):
        trial_no += 1
        w_trial = [random.choice(VALID_VALUES) for _ in range(TOTAL_LAYERS)]
        i_trial = [random.choice(VALID_VALUES) for _ in range(TOTAL_LAYERS)]

        print(f"\n  [{trial_no:03d}] Random trial")
        print(f"    W: {w_trial}")
        print(f"    I: {i_trial}")

        update_c_source(w_trial, i_trial)
        if not build_project():
            continue
        map_val, _ = run_test()
        if map_val is None:
            continue

        entry = {
            "trial": trial_no, "type": "random",
            "map": map_val,
            "weight": w_trial, "input": i_trial,
            "timestamp": datetime.now().isoformat()
        }
        log_entries.append(entry)
        save_log(log_entries)

        if map_val > best_map:
            best_map = map_val
            best_weight = list(w_trial)
            best_input  = list(i_trial)
            save_best(best_map, best_weight, best_input)

    return best_map, best_weight, best_input, log_entries

# ============================================================
# 빠른 테스트 모드: 단일 설정 검증
# ============================================================
def single_test(weight_mults, input_mults):
    """단일 설정 실행 (설정값 확인용)"""
    print("\n  단일 설정 테스트")
    print(f"  Weight: {weight_mults}")
    print(f"  Input:  {input_mults}")
    update_c_source(weight_mults, input_mults)
    if not build_project():
        return None
    map_val, output = run_test()
    return map_val

# ============================================================
# 현재 C 소스에서 multiplier 값 읽기
# ============================================================
def read_current_multipliers():
    """현재 C 파일에서 multiplier 값 추출"""
    with open(C_SRC_FILE, 'r') as f:
        content = f.read()

    def extract_array(pattern):
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None
        nums = re.findall(r'\b(\d+)\b', match.group(1))
        return [int(n) for n in nums if n in [str(v) for v in VALID_VALUES + [1, 2, 4, 8]]]

    w_match = re.search(
        r'float weight_quant_multiplier\[TOTAL_CALIB_LAYER\]\s*=\s*\{([^}]*)\}', 
        content, re.DOTALL
    )
    i_match = re.search(
        r'float input_quant_multiplier\[TOTAL_CALIB_LAYER\]\s*=\s*\{([^}]*)\}', 
        content, re.DOTALL
    )

    if w_match and i_match:
        w_nums = re.findall(r'\b(\d+)\b', w_match.group(1))
        i_nums = re.findall(r'\b(\d+)\b', i_match.group(1))
        # 숫자만 추출 (주석 숫자 제거: conv 번호는 대체로 한 자리)
        w_vals = [int(n) for n in w_nums if int(n) >= 16][:TOTAL_LAYERS]
        i_vals = [int(n) for n in i_nums if int(n) >= 16][:TOTAL_LAYERS]
        return w_vals, i_vals
    return None, None

# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("  AIX2026 Quantization Multiplier Auto-Tuner")
    print(f"  시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 경로 확인
    if not os.path.exists(C_SRC_FILE):
        print(f"[ERROR] C 소스 파일을 찾을 수 없음: {C_SRC_FILE}")
        sys.exit(1)
    if not os.path.exists(SKELETON_DIR):
        print(f"[ERROR] Skeleton 디렉토리를 찾을 수 없음: {SKELETON_DIR}")
        sys.exit(1)

    # 현재 설정 읽기
    cur_w, cur_i = read_current_multipliers()
    if cur_w and len(cur_w) == TOTAL_LAYERS:
        print(f"\n  현재 C 파일의 Weight multipliers: {cur_w}")
        print(f"  현재 C 파일의 Input  multipliers: {cur_i}")
        start_w = cur_w
        start_i = cur_i
    else:
        print(f"\n  자동 파싱 실패 → 베이스라인 값 사용")
        start_w = BASELINE_WEIGHT
        start_i = BASELINE_INPUT

    # 실행 모드 선택
    mode = sys.argv[1] if len(sys.argv) > 1 else "greedy"

    if mode == "greedy":
        # 기본 모드: Greedy Hill-Climbing
        best_map, best_w, best_i, log = greedy_search(start_w, start_i)

    elif mode == "random":
        # 랜덤 탐색 (n_trials 조정 가능)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        best_map, best_w, best_i, log = random_search(start_w, start_i, BASELINE_MAP, n_trials=n)

    elif mode == "both":
        # Greedy 후 Random 추가 탐색
        best_map, best_w, best_i, log1 = greedy_search(start_w, start_i)
        best_map, best_w, best_i, log2 = random_search(best_w, best_i, best_map, n_trials=100)
        log = log1 + log2

    elif mode == "test":
        # 현재 설정만 테스트
        map_val = single_test(start_w, start_i)
        print(f"\n  결과 mAP: {map_val}")
        sys.exit(0)

    else:
        print(f"  사용법: python aix_autotuner.py [greedy|random|both|test] [n_random_trials]")
        sys.exit(1)

    # 최종 요약
    print("\n" + "="*60)
    print("  최적화 완료")
    print(f"  최고 mAP:  {best_map:.4f}%  (베이스라인: {BASELINE_MAP:.2f}%)")
    print(f"  개선폭:    +{best_map - BASELINE_MAP:.4f}%")
    print(f"  최적 Weight: {best_w}")
    print(f"  최적 Input:  {best_i}")
    print(f"  로그 파일:   {LOG_FILE}")
    print(f"  최적 파일:   {BEST_FILE}")
    print("="*60)