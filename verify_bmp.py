from PIL import Image
import numpy as np
import os

def merge_bmp_channels(bmp_dir, output_filename, num_channels=4, width=64, height=64):
    print(f"--- Merging {num_channels} BMP channels ---")
    
    # 합성된 이미지를 저장할 빈 배열 (float 타입으로 시작하여 정밀도 유지)
    merged_data = np.zeros((height, width), dtype=np.float32)
    valid_channels = 0

    for ch in range(num_channels):
        bmp_filename = f"CONV04_output_ch{ch:02d}.bmp"
        bmp_path = os.path.join(bmp_dir, bmp_filename)

        if not os.path.exists(bmp_path):
            print(f"Skip: {bmp_filename} not found.")
            continue

        # 1. BMP 이미지 로드 및 그레이스케일 변환
        img = Image.open(bmp_path).convert('L') # 'L' 모드는 8비트 그레이스케일
        img_data = np.array(img, dtype=np.float32)

        # 2. 이미지 데이터를 누적해서 더함
        merged_data += img_data
        valid_channels += 1
        print(f"  Loaded {bmp_filename}")

    if valid_channels == 0:
        print("Error: No BMP files found to merge.")
        return

    # 3. 평균 계산 (픽셀 값이 255를 넘지 않도록 채널 수로 나눔)
    merged_data /= valid_channels

    # 4. 데이터를 다시 8비트 정수(0~255)로 변환
    final_data = merged_data.astype(np.uint8)

    # 5. 결과 이미지 저장
    final_img = Image.fromarray(final_data, mode='L')
    final_img.save(output_filename)
    print(f"Success: Merged image saved as {output_filename}")

# --- 설정부 (사용자 환경에 맞게 수정) ---
# BMP 파일이 들어있는 폴더를 지정하세요.
BMP_DIR = "." 
OUTPUT_FILE = os.path.join(BMP_DIR, "CONV04_merged_feature_map.bmp")

# 함수 실행
# Pillow 라이브러리가 없다면 'pip install Pillow'로 설치해야 합니다.
try:
    merge_bmp_channels(BMP_DIR, OUTPUT_FILE)
except ImportError:
    print("Error: Pillow library is not installed. Run 'pip install Pillow' first.")