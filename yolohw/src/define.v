//`define PRELOAD
//`define DEBUG
`define NUM_BRAMS   16
`define BRAM_WIDTH  128
`define BRAM_DELAY  3
// `define FPGA    1   // 시뮬레이션 시 주석 유지

// =============================================================
// LAYER 선택 가이드:
//   CONV00: Ni=3,  No=16, W=256, H=256
//   CONV02: Ni=16, No=32, W=128, H=128
//   CONV04: Ni=32, No=64, W=64,  H=64
// =============================================================

// 현재 레이어: CONV00
parameter IFM_WIDTH        = 256;   // CONV00:256 / CONV02:128 / CONV04:64
parameter IFM_HEIGHT       = 256;   // CONV00:256 / CONV02:128 / CONV04:64
parameter IFM_CHANNEL      = 3;
parameter IFM_DATA_SIZE    = IFM_HEIGHT*IFM_WIDTH*2;
parameter IFM_WORD_SIZE    = 32/2;
parameter IFM_DATA_SIZE_32 = IFM_HEIGHT*IFM_WIDTH;   // 채널0만 (bmp 시각화용)
parameter IFM_WORD_SIZE_32 = 32;
parameter Fx = 3, Fy = 3;
parameter Ni = 3,  No = 16;         // CONV00:3,16 / CONV02:16,32 / CONV04:32,64
parameter WGT_DATA_SIZE    = Fx*Fy*Ni*No;
parameter WGT_WORD_SIZE    = 32;

// [V2] 전체 Ni채널 입력 hex 파일 (Ni * H * W 개 8비트 픽셀)
parameter IFM_FILE_ALL = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_feamap/CONV00_input.hex";
// 채널0 32b 포맷 (bmp 시각화용)
parameter IFM_FILE_32  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_feamap/CONV00_input_32b.hex";
parameter IFM_FILE     = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_feamap/CONV00_input_16b.hex";
parameter WGT_FILE     = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_param/CONV00_param_weight.hex";

parameter CONV_INPUT_IMG00  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_input_ch00.bmp";
parameter CONV_INPUT_IMG01  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_input_ch01.bmp";
parameter CONV_INPUT_IMG02  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_input_ch02.bmp";
parameter CONV_INPUT_IMG03  = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_input_ch03.bmp";

parameter CONV_OUTPUT_IMG00 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch00.bmp";
parameter CONV_OUTPUT_IMG01 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch01.bmp";
parameter CONV_OUTPUT_IMG02 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch02.bmp";
parameter CONV_OUTPUT_IMG03 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch03.bmp";
parameter CONV_OUTPUT_IMG04 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch04.bmp";
parameter CONV_OUTPUT_IMG05 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch05.bmp";
parameter CONV_OUTPUT_IMG06 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch06.bmp";
parameter CONV_OUTPUT_IMG07 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch07.bmp";
parameter CONV_OUTPUT_IMG08 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch08.bmp";
parameter CONV_OUTPUT_IMG09 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch09.bmp";
parameter CONV_OUTPUT_IMG10 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch10.bmp";
parameter CONV_OUTPUT_IMG11 = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_hw/CONV00_output_ch11.bmp";

parameter DW         = 32;
parameter AW         = 16;
parameter DEPTH      = 65536;
parameter N_DELAY    = 1;
parameter BUFF_WIDTH = 32;
parameter BUFF_DEPTH = 4096;
parameter BUFF_ADDR_W = $clog2(BUFF_DEPTH);
//`define CHECK_DMA_WRITE 1
