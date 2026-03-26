# AIX2026 Deep Learning Hardware Accelerator Design Contest
## Report 2: Mid-term Evaluation — RTL Simulation for CONV00, CONV02, CONV04

---

## 📁 프로젝트 구조

```
DeepLearning-Hardware-Architecture-Contest/
├── hex/                          # 시뮬레이션용 hex 파일 (C코드로 생성)
│   ├── CONV00_input.hex          # CONV00 입력 IFM (planar INT8)
│   ├── CONV00_param_weight.hex   # CONV00 양자화 가중치 (TI=4 packed)
│   ├── CONV00_param_biases.hex   # CONV00 양자화 바이어스 (INT16)
│   ├── CONV02_param_weight.hex
│   ├── CONV02_param_biases.hex
│   ├── CONV04_param_weight.hex
│   ├── CONV04_param_biases.hex
│   ├── CONV00_output.hex         # SW 기준 출력 (HW 검증용)
│   ├── CONV02_output.hex
│   └── CONV04_output.hex
├── skeleton/                     # C++ 소프트웨어 (darknet 기반)
│   └── src/
│       └── yolov2_forward_network_quantized.c   # INT8 양자화 추론 + hex 생성
└── yolohw/
    ├── src/                      # Design Sources (Verilog)
    │   ├── yolo_engine.v         # ★ 메인 가속기 엔진 (수정됨)
    │   ├── mac.v                 # MAC 유닛 (8b×8b × 16개)
    │   ├── adder_tree.v          # 4단계 Adder Tree
    │   ├── mul.v                 # 8비트 곱셈기 (DSP/LUT 선택)
    │   ├── cnn_ctrl.v            # FSM 기반 CNN 컨트롤러
    │   ├── axi_dma_ctrl.v        # AXI DMA 제어기
    │   ├── axi_dma_rd.v          # AXI DMA Read
    │   ├── axi_dma_wr.v          # AXI DMA Write
    │   ├── dpram_wrapper.v       # Dual-port RAM 래퍼
    │   ├── spram_wrapper.v       # Single-port RAM 래퍼
    │   ├── axi_sram_if.v         # AXI-SRAM 인터페이스
    │   ├── axi_slave_if_sync.v   # AXI Slave 동기화
    │   ├── sram.v                # SRAM 모델
    │   ├── sram_ctrl.v           # SRAM 컨트롤러
    │   ├── sync_reg_fifo.v       # 동기 FIFO
    │   ├── bmp_image_writer.v    # BMP 이미지 저장 (시뮬레이션용)
    │   ├── user_param_h.v        # ★ 파라미터 헤더 (수정됨)
    │   ├── user_define_h.v       # 매크로 정의 헤더
    │   └── define.v              # 전역 파라미터 정의
    └── sim/
        ├── yolo_engine_tb.v      # ★ 테스트벤치 (수정됨)
        └── sim_dram_model/       # DRAM 시뮬레이션 모델
```

---

## ⚙️ 요구 환경

- **Xilinx Vivado 2021.1** 이상 (2025.2 확인됨)
- **Target FPGA**: Nexys A7 — `xc7a100tcsg324-1`
- **C 컴파일러**: Windows (Visual Studio 2019) 또는 Linux (gcc)

---

## 🚀 시뮬레이션 실행 방법

### Step 1: 저장소 클론

```bash
git clone https://github.com/SeungYunChoi0/DeepLearning-Hardware-Architecture-Contest.git
cd DeepLearning-Hardware-Architecture-Contest
```

### Step 2: Vivado 프로젝트 생성

Vivado를 열고 새 프로젝트를 생성합니다.
- Part: `xc7a100tcsg324-1` (Nexys A7 기준)

### Step 3: Design Sources 추가

**Sources 패널 → Add Sources → Add or create design sources** 에서 아래 파일을 모두 추가합니다.

```
yolohw/src/yolo_engine.v
yolohw/src/mac.v
yolohw/src/adder_tree.v
yolohw/src/mul.v
yolohw/src/cnn_ctrl.v
yolohw/src/axi_dma_ctrl.v
yolohw/src/axi_dma_rd.v
yolohw/src/axi_dma_wr.v
yolohw/src/dpram_wrapper.v
yolohw/src/spram_wrapper.v
yolohw/src/axi_sram_if.v
yolohw/src/axi_slave_if_sync.v
yolohw/src/sram.v
yolohw/src/sram_ctrl.v
yolohw/src/sync_reg_fifo.v
yolohw/src/bmp_image_writer.v
yolohw/src/user_param_h.v      ← Verilog Header로 설정 필요
yolohw/src/user_define_h.v     ← Verilog Header로 설정 필요
yolohw/src/define.v            ← Verilog Header로 설정 필요
```

> **⚠️ 주의**: `yolo_engine_ip.v`, `yolo_engine_axi.v`는 추가하지 마세요. 최종 FPGA 구현 단계에서만 사용합니다.

### Step 4: Verilog Header 설정

Tcl 콘솔에서 아래 명령을 실행합니다 (한 번만 실행하면 프로젝트에 저장됨):

```tcl
foreach f {user_param_h.v user_define_h.v define.v} {
    set_property file_type {Verilog Header} [get_files $f]
}
```

### Step 5: Top Module 설정

```tcl
set_property top yolo_engine [current_fileset]
```

### Step 6: Simulation Sources 추가

**Sources 패널 → Add Sources → Add or create simulation sources** 에서 추가:

```
# TB 파일
yolohw/sim/yolo_engine_tb.v

# hex 파일 (필수 - 연산용)
hex/CONV00_input.hex
hex/CONV00_param_weight.hex
hex/CONV00_param_biases.hex
hex/CONV02_param_weight.hex
hex/CONV02_param_biases.hex
hex/CONV04_param_weight.hex
hex/CONV04_param_biases.hex

# hex 파일 (선택 - SW vs HW 검증용)
hex/CONV00_output.hex
hex/CONV02_output.hex
hex/CONV04_output.hex
```

### Step 7: Simulation Top 설정

```tcl
set_property top yolo_engine_tb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
```

### Step 8: 시뮬레이션 실행

**Flow Navigator → Run Simulation → Run Behavioral Simulation**

또는 Tcl 콘솔:
```tcl
launch_simulation
run all
```

### Step 9: 결과 확인

시뮬레이션 완료 후 Tcl 콘솔에서 아래 메시지를 확인합니다:

```
[V4] 로드 완료
[V4] IFM[0] R=xxx G=xxx B=xxx
[V4] CONV00 시작 (256x256, Ni=3, No=16, shift>>7)
[V4] Layer0 CONV+Bias+ReLU 완료 -> MaxPool
[V4] Layer0 MaxPool 완료
[V4] Layer1 시작
...
[V4] 완료: CONV00->Pool->CONV02->Pool->CONV04->Pool
[TB] network_done!
[VERIFY] CONV00 OFM: total=1048576, errors=0 -> PASS
[VERIFY] CONV02 Pool: total=262144, errors=0 -> PASS
[VERIFY] CONV04 Pool: total=131072, errors=0 -> PASS
```

BMP 이미지 출력 경로 (`user_param_h.v`에서 설정):
```
yolohw/sim/inout_data_hw/CONV00_output_ch00.bmp  (256x256)
yolohw/sim/inout_data_hw/CONV00_output_ch01.bmp
yolohw/sim/inout_data_hw/CONV00_output_ch02.bmp
yolohw/sim/inout_data_hw/CONV00_output_ch03.bmp
yolohw/sim/inout_data_hw/CONV02_pool_ch00.bmp    (128x128)
yolohw/sim/inout_data_hw/CONV02_pool_ch01.bmp
yolohw/sim/inout_data_hw/CONV04_pool_ch00.bmp    (64x64)
yolohw/sim/inout_data_hw/CONV04_pool_ch01.bmp
```

---

## 📝 C 코드로 hex 파일 재생성하는 방법

`hex/` 폴더의 파일들은 `skeleton/src/yolov2_forward_network_quantized.c`를 실행하여 생성됩니다.

### Windows (Visual Studio)
```
skeleton/ 폴더의 yolo_cpu.sln 열기 → Build → Run
결과: skeleton/bin/log_feamap/, skeleton/bin/log_param/ 에 생성
```

### Mac/Linux
```bash
cd skeleton/
make
cd bin/
sh script-unix-aix2024-test-one-quantized.sh
```

생성된 hex 파일을 `hex/` 폴더로 복사하면 됩니다.

---

## 🔢 양자화 파라미터 요약

| Layer | input_mult | weight_mult | shift | bias_mult |
|-------|-----------|-------------|-------|-----------|
| CONV00 | 128 | 32 | 7 | 4096 |
| CONV02 | 32 | 256 | 8 | 8192 |
| CONV04 | 32 | 256 | 9 | 8192 |

mAP: **79.28%** (baseline 81.76% 대비 96.96%)
