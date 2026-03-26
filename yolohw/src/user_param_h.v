// -------------------------------------------------------------
// user_param_h.v
// yolo_engine.v / yolo_engine_tb.v (V4) 대응
// CONV00 -> MaxPool -> CONV02 -> MaxPool -> CONV04 -> MaxPool
//
// IMPORTANT:
//   sim_1 Add Sources에 아래 hex 파일을 추가할 것:
//     CONV00_input.hex
//     CONV00_param_weight.hex / CONV02_param_weight.hex / CONV04_param_weight.hex
//     CONV00_param_biases.hex / CONV02_param_biases.hex / CONV04_param_biases.hex
//
//   bmp 출력 폴더가 없으면 미리 생성:
//     yolohw/sim/inout_data_hw/
// -------------------------------------------------------------

// -------------------------------------------------------------
// IFM 기본 크기 (CONV00 입력 기준)
// -------------------------------------------------------------
parameter IFM_WIDTH         = 256;
parameter IFM_HEIGHT        = 256;
parameter IFM_CHANNEL       = 3;

// -------------------------------------------------------------
// TB sram 모델용 IFM 파일 (더미 - yolo_engine.v에서 직접 로드)
// IFM_FILE_16b -> IFM_FILE 로 통일 (CONV00_input.hex 사용)
// -------------------------------------------------------------
parameter IFM_FILE          = "CONV00_input.hex";

// -------------------------------------------------------------
// SRAM / Buffer 크기
// -------------------------------------------------------------
parameter DW            = 32;
parameter AW            = 16;
parameter DEPTH         = 65536;
parameter N_DELAY       = 1;

parameter BUFF_WIDTH    = 32;
parameter BUFF_DEPTH    = 4096;
parameter BUFF_ADDR_W   = $clog2(BUFF_DEPTH);  // = 12

// -------------------------------------------------------------
// 입력 이미지 bmp (디버그용 - CHECK_DMA_WRITE define 시 사용)
// -------------------------------------------------------------
parameter CONV_INPUT_IMG00 = "./CONV00_input_ch00.bmp";
parameter CONV_INPUT_IMG01 = "./CONV00_input_ch01.bmp";
parameter CONV_INPUT_IMG02 = "./CONV00_input_ch02.bmp";
parameter CONV_INPUT_IMG03 = "./CONV00_input_ch03.bmp";

// -------------------------------------------------------------
// 출력 bmp 파일 경로
//
// IMG00~03 : CONV00 출력 (ofm_buf, 256x256, ch0~3)
// IMG04~05 : CONV02 MaxPool 후 (pool_buf, 128x128, ch0~1)
// IMG06~07 : CONV04 MaxPool 후 (pool_buf,  64x64,  ch0~1)
// -------------------------------------------------------------
// CONV00 출력 (256x256)
parameter CONV_OUTPUT_IMG00 = "./CONV00_output_ch00.bmp";
parameter CONV_OUTPUT_IMG01 = "./CONV00_output_ch01.bmp";
parameter CONV_OUTPUT_IMG02 = "./CONV00_output_ch02.bmp";
parameter CONV_OUTPUT_IMG03 = "./CONV00_output_ch03.bmp";

// CONV02 MaxPool 후 (128x128)
parameter CONV_OUTPUT_IMG04 = "./CONV02_pool_ch00.bmp";
parameter CONV_OUTPUT_IMG05 = "./CONV02_pool_ch01.bmp";

// CONV04 MaxPool 후 (64x64)
parameter CONV_OUTPUT_IMG06 = "./CONV04_pool_ch00.bmp";
parameter CONV_OUTPUT_IMG07 = "./CONV04_pool_ch01.bmp";