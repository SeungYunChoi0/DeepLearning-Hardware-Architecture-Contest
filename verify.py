import os

def compare_yolo_hex(golden_path, hw_out_dir, num_channels=4, width=64, height=64):
    pixel_per_channel = width * height
    
    # 1. 정답지(Golden) 로드
    if not os.path.exists(golden_path):
        print(f"Error: Golden file not found at {golden_path}")
        return

    with open(golden_path, 'r') as f:
        # 공백/줄바꿈 제거 후 리스트 저장
        golden_data = [line.strip().lower() for line in f.readlines() if line.strip()]

    print(f"--- CONV04 Verification Start ---")
    print(f"Golden Data Total Length: {len(golden_data)} (Expected for 64ch: {pixel_per_channel * 64})")

    for ch in range(num_channels):
        hw_filename = f"CONV04_hw_out_ch{ch:02d}.hex"
        hw_path = os.path.join(hw_out_dir, hw_filename)

        if not os.path.exists(hw_path):
            print(f"Skip: {hw_filename} not found.")
            continue

        # 2. 하드웨어 결과 로드
        with open(hw_path, 'r') as f:
            hw_data = [line.strip().lower() for line in f.readlines() if line.strip()]

        # 3. 정답지에서 해당 채널 구간 추출
        start_idx = ch * pixel_per_channel
        end_idx = start_idx + pixel_per_channel
        current_golden_ch = golden_data[start_idx:end_idx]

        # 4. 비교 연산
        matches = 0
        mismatch_count = 0
        total_pixels = len(hw_data)

        for i in range(total_pixels):
            # 16진수 문자열 직접 비교 (예: "1a" == "1a")
            if i < len(current_golden_ch) and hw_data[i] == current_golden_ch[i]:
                matches += 1
            else:
                mismatch_count += 1
                if mismatch_count <= 5: # 처음 5개 틀린 지점만 출력
                    g_val = current_golden_ch[i] if i < len(current_golden_ch) else "EOF"
                    print(f"  [CH{ch}] Mismatch at pixel {i}: Golden={g_val}, HW={hw_data[i]}")

        accuracy = (matches / pixel_per_channel) * 100
        print(f"Result Channel {ch:02d}: Matches={matches}/{pixel_per_channel} ({accuracy:.2f}%)")

# --- 설정부 (사용자 환경에 맞게 수정) ---
# 정답지가 들어있는 폴더와 하드웨어 출력이 나오는 폴더를 지정하세요.
GOLDEN_FILE = "/Volumes/Yun_ssd/AIX2026/AIX/hex/CONV04_output.hex"
HW_OUT_DIR  = "/Volumes/Yun_ssd/AIX2026/AIX"

compare_yolo_hex(GOLDEN_FILE, HW_OUT_DIR)