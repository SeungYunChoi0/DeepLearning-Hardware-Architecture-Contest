// =============================================================
// LAYER 선택 가이드 (수정 완료: CONV04)
//   CONV00: Ni=3,  No=16, W=256, H=256
//   CONV02: Ni=16, No=32, W=128, H=128
//   CONV04: Ni=32, No=64, W=64,  H=64 
// =============================================================

// 현재 레이어: CONV04 설정
parameter IFM_WIDTH        = 64;  // [cite: 167-168]
parameter IFM_HEIGHT       = 64;  // [cite: 169]
parameter IFM_CHANNEL      = 32;  // Ni와 동일

parameter IFM_DATA_SIZE    = IFM_HEIGHT*IFM_WIDTH*2;
parameter IFM_WORD_SIZE    = 32/2;
parameter IFM_DATA_SIZE_32 = IFM_HEIGHT*IFM_WIDTH;

// 채널0만 (bmp 시각화용)
parameter IFM_WORD_SIZE_32 = 32;
parameter Fx = 3, Fy = 3;

// CONV04 채널 설정: 입력(Ni)=32, 출력(No)=64 
parameter Ni = 32; 
parameter No = 64; 
parameter WGT_DATA_SIZE    = Fx*Fy*Ni*No; // 3*3*32*64 
parameter WGT_WORD_SIZE    = 32;

// [V2] 전체 Ni채널 입력 hex 파일 (CONV04용으로 변경) [cite: 173-175]
parameter IFM_FILE_ALL = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_feamap/CONV04_input.hex";
parameter IFM_FILE_32  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_feamap/CONV04_input_32b.hex";
parameter IFM_FILE     = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_feamap/CONV04_input_16b.hex";
parameter WGT_FILE     = "C:/Users/15Z980/Desktop/yun/yolohw/sim/log_param/CONV04_param_weight.hex";

// 입력 이미지 디버그 bmp 경로 (CONV04) [cite: 175-176]
parameter CONV_INPUT_IMG00  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_input_ch00.bmp";
parameter CONV_INPUT_IMG01  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_input_ch01.bmp";
parameter CONV_INPUT_IMG02  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_input_ch02.bmp";
parameter CONV_INPUT_IMG03  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_input_ch03.bmp";

// 출력 결과 bmp 경로 (CONV04) [cite: 176-178]
parameter CONV_OUTPUT_IMG00 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_output_ch00.bmp";
parameter CONV_OUTPUT_IMG01 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_output_ch01.bmp";
parameter CONV_OUTPUT_IMG02 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_output_ch02.bmp";
parameter CONV_OUTPUT_IMG03 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV04_output_ch03.bmp";
// ... (필요에 따라 IMG11까지 확장 가능)

parameter DW         = 32;
parameter AW         = 16;
parameter DEPTH      = 65536;
parameter N_DELAY    = 1;
parameter BUFF_WIDTH = 32;
parameter BUFF_DEPTH = 4096;
parameter BUFF_ADDR_W = $clog2(BUFF_DEPTH);