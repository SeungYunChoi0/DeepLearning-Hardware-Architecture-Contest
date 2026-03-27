// =============================================================
// CONV04 전용 설정 파일 (최종 경로 및 규격 반영)
// =============================================================

// 1. 이미지 및 채널 규격 [cite: 167-169, 172]
parameter IFM_WIDTH        = 64;  
parameter IFM_HEIGHT       = 64;  
parameter Ni               = 32; 
parameter No               = 64; 
parameter Fx               = 3;
parameter Fy               = 3;

// 2. 데이터 사이즈 계산 
// _input.hex용 (전체 8비트 데이터 개수: 64*64*32)
parameter IFM_DATA_SIZE    = IFM_HEIGHT * IFM_WIDTH * Ni; 

// _input_32b.hex용 (4채널당 1줄 패킹: 64*64*8)
// CONV04는 Ni=32이므로 한 픽셀당 8줄(32/4)이 들어있습니다. [cite: 140-142]
parameter IFM_DATA_SIZE_32 = IFM_HEIGHT * IFM_WIDTH * (Ni/4); 

parameter IFM_WORD_SIZE_32 = 32;
parameter WGT_DATA_SIZE    = Fx * Fy * Ni * No;
parameter WGT_WORD_SIZE    = 32;

// 3. 파일 경로 (사용자가 옮겨둔 새 경로로 통일) 
parameter HEX_PATH     = "C:/Users/15Z980/Desktop/yun/DeepLearning-Hardware-Architecture-Contest/hex";

parameter IFM_FILE_ALL = {HEX_PATH, "/CONV04_input.hex"};
parameter IFM_FILE_32  = {HEX_PATH, "/CONV04_input_32b.hex"};
parameter IFM_FILE     = {HEX_PATH, "/CONV04_input_16b.hex"};
parameter WGT_FILE     = {HEX_PATH, "/CONV04_param_weight.hex"};

// 4. 출력 이미지 저장 경로 (기존 경로 유지) [cite: 175-178]
parameter OUT_PATH     = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw";

parameter CONV_OUTPUT_IMG00 = {OUT_PATH, "/CONV04_output_ch00.bmp"};
parameter CONV_OUTPUT_IMG01 = {OUT_PATH, "/CONV04_output_ch01.bmp"};
parameter CONV_OUTPUT_IMG02 = {OUT_PATH, "/CONV04_output_ch02.bmp"};
parameter CONV_OUTPUT_IMG03 = {OUT_PATH, "/CONV04_output_ch03.bmp"};

// 기타 하드웨어 파라미터 [cite: 179-182]
parameter DW         = 32;
parameter AW         = 16;
parameter DEPTH      = 65536;
parameter BUFF_DEPTH = 4096;