//----------------------------------------------------------------+
// yolo_engine.v (V4 - 수정본)
// 완전한 파이프라인:
//   IFM -> CONV(MAC누산) -> +Bias -> ReLU -> >>shift -> MaxPool
//
// [수정사항]
//   1. ifm_buf 크기: 262144 (16x128x128 커버)
//   2. 가중치 로드: TI=4 패킹 언패킹 + CONV00_input.hex 직접 로드
//   3. MaxPool: 지연 레지스터 제거 -> 1클럭 지역변수 방식
//
// V2 C코드 기준 레이어별 파라미터:
//   CONV00: in_m=128, w_m=32,  next_m=32 -> shift>>7,  bias_mult=4096
//   CONV02: in_m=32,  w_m=256, next_m=32 -> shift>>8,  bias_mult=8192
//   CONV04: in_m=32,  w_m=256, next_m=16 -> shift>>9,  bias_mult=8192
//
// sim_1에 Add Sources로 추가할 hex 파일:
//   CONV00_input.hex
//   CONV00/02/04_param_weight.hex  (TI=4 packed 32bit)
//   CONV00/02/04_param_biases.hex  (16bit signed)
//
// 하드웨어: TO=12, DSP=192개(80%), mac×12
//----------------------------------------------------------------+
`timescale 1ns/1ps

module yolo_engine #(
    parameter AXI_WIDTH_AD   = 32,
    parameter AXI_WIDTH_ID   = 4,
    parameter AXI_WIDTH_DA   = 32,
    parameter AXI_WIDTH_DS   = AXI_WIDTH_DA/8,
    parameter OUT_BITS_TRANS = 18,
    parameter WBUF_AW        = 9,
    parameter WBUF_DW        = 8*3*3*16,
    parameter WBUF_DS        = WBUF_DW/8,
    parameter MEM_BASE_ADDR      = 'h8000_0000,
    parameter MEM_DATA_BASE_ADDR = 4096
)(
    input                        clk,
    input                        rstn,
    input  [31:0]                i_ctrl_reg0,
    input  [31:0]                i_ctrl_reg1,
    input  [31:0]                i_ctrl_reg2,
    input  [31:0]                i_ctrl_reg3,
    output                       M_ARVALID,
    input                        M_ARREADY,
    output [AXI_WIDTH_AD-1:0]    M_ARADDR,
    output [AXI_WIDTH_ID-1:0]    M_ARID,
    output [7:0]                 M_ARLEN,
    output [2:0]                 M_ARSIZE,
    output [1:0]                 M_ARBURST,
    output [1:0]                 M_ARLOCK,
    output [3:0]                 M_ARCACHE,
    output [2:0]                 M_ARPROT,
    output [3:0]                 M_ARQOS,
    output [3:0]                 M_ARREGION,
    output [3:0]                 M_ARUSER,
    input                        M_RVALID,
    output                       M_RREADY,
    input  [AXI_WIDTH_DA-1:0]    M_RDATA,
    input                        M_RLAST,
    input  [AXI_WIDTH_ID-1:0]    M_RID,
    input  [3:0]                 M_RUSER,
    input  [1:0]                 M_RRESP,
    output                       M_AWVALID,
    input                        M_AWREADY,
    output [AXI_WIDTH_AD-1:0]    M_AWADDR,
    output [AXI_WIDTH_ID-1:0]    M_AWID,
    output [7:0]                 M_AWLEN,
    output [2:0]                 M_AWSIZE,
    output [1:0]                 M_AWBURST,
    output [1:0]                 M_AWLOCK,
    output [3:0]                 M_AWCACHE,
    output [2:0]                 M_AWPROT,
    output [3:0]                 M_AWQOS,
    output [3:0]                 M_AWREGION,
    output [3:0]                 M_AWUSER,
    output                       M_WVALID,
    input                        M_WREADY,
    output [AXI_WIDTH_DA-1:0]    M_WDATA,
    output [AXI_WIDTH_DS-1:0]    M_WSTRB,
    output                       M_WLAST,
    output [AXI_WIDTH_ID-1:0]    M_WID,
    output [3:0]                 M_WUSER,
    input                        M_BVALID,
    output                       M_BREADY,
    input  [1:0]                 M_BRESP,
    input  [AXI_WIDTH_ID-1:0]    M_BID,
    input                        M_BUSER,
    output                       network_done,
    output                       network_done_led,
    output [AXI_WIDTH_DA-1:0]    read_data,
    output                       read_data_vld
);

`include "user_define_h.v"
`include "user_param_h.v"

//------------------------------------------------------------
// 하드웨어 파라미터
//------------------------------------------------------------
localparam TO          = 12;
localparam MAC_LATENCY = 9;
localparam NUM_LAYERS  = 3;

//------------------------------------------------------------
// 레이어 파라미터 함수
//------------------------------------------------------------
function [11:0] f_width;
    input [1:0] i;
    case(i) 2'd0:f_width=256; 2'd1:f_width=128; default:f_width=64; endcase
endfunction
function [11:0] f_height;
    input [1:0] i;
    case(i) 2'd0:f_height=256; 2'd1:f_height=128; default:f_height=64; endcase
endfunction
function [5:0] f_ni;
    input [1:0] i;
    case(i) 2'd0:f_ni=3; 2'd1:f_ni=16; default:f_ni=32; endcase
endfunction
function [6:0] f_no;
    input [1:0] i;
    case(i) 2'd0:f_no=16; 2'd1:f_no=32; default:f_no=64; endcase
endfunction
// descale shift: (in_m * w_m) / next_in_m = 128*32/32=128(>>7), 32*256/32=256(>>8), 32*256/16=512(>>9)
function [3:0] f_shift;
    input [1:0] i;
    case(i) 2'd0:f_shift=7; 2'd1:f_shift=8; default:f_shift=9; endcase
endfunction

//------------------------------------------------------------
// [수정1] 가중치 버퍼 - 8비트 개별 배열 (언패킹 후)
//   CONV00: filter_size=27, No=16  -> 27*16  = 432
//   CONV02: filter_size=144, No=32 -> 144*32 = 4608
//   CONV04: filter_size=288, No=64 -> 288*64 = 18432
//------------------------------------------------------------
reg [7:0] w0 [0:431  ];
reg [7:0] w1 [0:4607 ];
reg [7:0] w2 [0:18431];

//------------------------------------------------------------
// 바이어스 버퍼 (16비트 부호있는, C코드 %04x 출력)
//------------------------------------------------------------
reg signed [15:0] b0 [0:15];
reg signed [15:0] b1 [0:31];
reg signed [15:0] b2 [0:63];

//------------------------------------------------------------
// IFM / OFM / Pool 버퍼 (8비트, planar: ch*W*H)
// [수정2] ifm_buf: 262144 (16*128*128) 커버
//------------------------------------------------------------
reg [7:0] ifm_buf  [0:262143 ];   // 최대 16*128*128
reg [7:0] ofm_buf  [0:1048575];   // 최대 16*256*256
reg [7:0] pool_buf [0:262143 ];   // 최대 16*128*128

//------------------------------------------------------------
// 누산기 (32비트)
//------------------------------------------------------------
reg signed [31:0] accum [0:TO-1][0:65535];

//------------------------------------------------------------
// MAC 신호
//------------------------------------------------------------
reg         mac_vld_i;
reg [127:0] mac_win [0:TO-1];
reg [127:0] mac_din;
wire [19:0] mac_acc_o [0:TO-1];
wire        mac_vld_o [0:TO-1];

//------------------------------------------------------------
// cnn_ctrl 신호
//------------------------------------------------------------
reg  [11:0] ctrl_q_width, ctrl_q_height;
reg  [11:0] ctrl_q_vsync_delay, ctrl_q_hsync_delay;
reg  [24:0] ctrl_q_frame_size;
reg         ctrl_q_start;
wire        ctrl_data_run;
wire [11:0] ctrl_row, ctrl_col;
wire        ctrl_end_frame;

//------------------------------------------------------------
// FSM 상태
//------------------------------------------------------------
localparam SEQ_IDLE       = 4'd0;
localparam SEQ_CONV_START = 4'd1;
localparam SEQ_CONV       = 4'd2;
localparam SEQ_DRAIN      = 4'd3;
localparam SEQ_RELU       = 4'd4;
localparam SEQ_MAXPOOL    = 4'd5;
localparam SEQ_COPY       = 4'd6;
localparam SEQ_NEXT       = 4'd7;
localparam SEQ_DONE       = 4'd8;

reg [3:0]  seq_state;
reg [1:0]  layer_idx;
reg [3:0]  to_cnt;
reg [5:0]  ni_cnt;
reg [11:0] cur_width, cur_height;
reg [5:0]  cur_ni;
reg [6:0]  cur_no;
reg [3:0]  cur_shift;
reg [31:0] vld_pix_idx;
reg [31:0] relu_idx;
reg [11:0] pool_row, pool_col;
reg [6:0]  pool_ch;
reg [31:0] copy_idx, copy_total;
reg [3:0]  drain_cnt;
// RELU 계산용 임시 변수
reg signed [31:0] acc_bias;
reg [7:0]         relu_out_r;

//------------------------------------------------------------
// AXI DMA
//------------------------------------------------------------
localparam BIT_TRANS = BUFF_ADDR_W;

reg  ap_start, ap_done, interrupt;
reg  [31:0] dram_base_addr_rd, dram_base_addr_wr, reserved_register;

wire ctrl_read, read_done;
wire [AXI_WIDTH_AD-1:0] read_addr;
wire [AXI_WIDTH_DA-1:0] read_data;
wire                     read_data_vld;
wire [BIT_TRANS-1:0]    read_data_cnt;
wire ctrl_write_done, ctrl_write, write_done, indata_req_wr;
wire [BIT_TRANS-1:0]    write_data_cnt;
wire [AXI_WIDTH_AD-1:0] write_addr;
wire [AXI_WIDTH_DA-1:0] write_data;
wire [BIT_TRANS-1:0] num_trans      = 16;
wire [15:0] max_req_blk_idx         = (256*256)/16;

//------------------------------------------------------------
// [수정2] 초기화 - 가중치 언패킹 + IFM 직접 로드
//
// C코드 save_quantized_model 출력 포맷:
//   TI=4 패킹: [31:24]=w[i+3] [23:16]=w[i+2] [15:8]=w[i+1] [7:0]=w[i+0]
//   CONV00: 16필터 * ceil(27/4)=7 = 112 words
//   CONV02: 32필터 * ceil(144/4)=36 = 1152 words
//   CONV04: 64필터 * ceil(288/4)=72 = 4608 words
//
// C코드 save_input_hex 출력 포맷:
//   CONV00_input.hex: planar [CH][H*W] 순서, 8비트/줄
//------------------------------------------------------------
integer ii;
initial begin : LOAD_HEX
    // 임시 패킹 버퍼 (파일 포맷과 1:1 대응)
    reg [31:0] w0_p [0:111 ];   // 16 * 7  = 112  words
    reg [31:0] w1_p [0:1151];   // 32 * 36 = 1152 words
    reg [31:0] w2_p [0:4607];   // 64 * 72 = 4608 words

    $readmemh("CONV00_param_weight.hex", w0_p);
    $readmemh("CONV02_param_weight.hex", w1_p);
    $readmemh("CONV04_param_weight.hex", w2_p);

    // 언패킹: [7:0]=w[4i+0], [15:8]=w[4i+1], [23:16]=w[4i+2], [31:24]=w[4i+3]
    for(ii=0; ii<112; ii=ii+1) begin
        if(4*ii+0 < 432) w0[4*ii+0] = w0_p[ii][ 7: 0];
        if(4*ii+1 < 432) w0[4*ii+1] = w0_p[ii][15: 8];
        if(4*ii+2 < 432) w0[4*ii+2] = w0_p[ii][23:16];
        if(4*ii+3 < 432) w0[4*ii+3] = w0_p[ii][31:24];
    end
    for(ii=0; ii<1152; ii=ii+1) begin
        w1[4*ii+0] = w1_p[ii][ 7: 0];
        w1[4*ii+1] = w1_p[ii][15: 8];
        w1[4*ii+2] = w1_p[ii][23:16];
        w1[4*ii+3] = w1_p[ii][31:24];
    end
    for(ii=0; ii<4608; ii=ii+1) begin
        w2[4*ii+0] = w2_p[ii][ 7: 0];
        w2[4*ii+1] = w2_p[ii][15: 8];
        w2[4*ii+2] = w2_p[ii][23:16];
        w2[4*ii+3] = w2_p[ii][31:24];
    end

    // IFM 직접 로드 (planar INT8, C코드 save_input_hex 출력)
    $readmemh("CONV00_input.hex", ifm_buf, 0, 196607);

    // 바이어스 로드 (16비트 signed)
    $readmemh("CONV00_param_biases.hex", b0);
    $readmemh("CONV02_param_biases.hex", b1);
    $readmemh("CONV04_param_biases.hex", b2);

    $display("[V4] 로드 완료");
    $display("[V4] IFM[0] R=%0d G=%0d B=%0d",
             $signed(ifm_buf[0]), $signed(ifm_buf[65536]), $signed(ifm_buf[131072]));
    $display("[V4] bias0[0]=%0d bias0[1]=%0d",
             $signed(b0[0]), $signed(b0[1]));
    $display("[V4] w0[0]=%0d w0[1]=%0d w0[26]=%0d",
             $signed(w0[0]), $signed(w0[1]), $signed(w0[26]));
end

//------------------------------------------------------------
// 제어
//------------------------------------------------------------
assign network_done     = interrupt;
assign network_done_led = interrupt;
always @(*) ap_done = ctrl_write_done;

always @(posedge clk, negedge rstn) begin
    if(~rstn) ap_start <= 0;
    else begin
        if(!ap_start && i_ctrl_reg0[0]) ap_start <= 1;
        else if(ap_done)                ap_start <= 0;
    end
end

always @(posedge clk, negedge rstn) begin
    if(~rstn) interrupt <= 0;
    else begin
        if(i_ctrl_reg0[0])             interrupt <= 0;
        else if(seq_state==SEQ_DONE)   interrupt <= 1;
    end
end

always @(posedge clk, negedge rstn) begin
    if(~rstn) begin
        dram_base_addr_rd<=0; dram_base_addr_wr<=0; reserved_register<=0;
    end else begin
        if(!ap_start && i_ctrl_reg0[0]) begin
            dram_base_addr_rd<=i_ctrl_reg1;
            dram_base_addr_wr<=i_ctrl_reg2;
            reserved_register<=i_ctrl_reg3;
        end else if(ap_done) begin
            dram_base_addr_rd<=0; dram_base_addr_wr<=0; reserved_register<=0;
        end
    end
end

//------------------------------------------------------------
// cnn_ctrl
//------------------------------------------------------------
cnn_ctrl u_cnn_ctrl(
    .clk(clk), .rstn(rstn),
    .q_width(ctrl_q_width), .q_height(ctrl_q_height),
    .q_vsync_delay(ctrl_q_vsync_delay),
    .q_hsync_delay(ctrl_q_hsync_delay),
    .q_frame_size(ctrl_q_frame_size),
    .q_start(ctrl_q_start),
    .o_ctrl_vsync_run(), .o_ctrl_vsync_cnt(),
    .o_ctrl_hsync_run(), .o_ctrl_hsync_cnt(),
    .o_ctrl_data_run(ctrl_data_run),
    .o_row(ctrl_row), .o_col(ctrl_col),
    .o_data_count(), .o_end_frame(ctrl_end_frame)
);

//------------------------------------------------------------
// MAC x TO=12
//------------------------------------------------------------
mac u_mac_00(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 0]),.din(mac_din),.acc_o(mac_acc_o[ 0]),.vld_o(mac_vld_o[ 0]));
mac u_mac_01(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 1]),.din(mac_din),.acc_o(mac_acc_o[ 1]),.vld_o(mac_vld_o[ 1]));
mac u_mac_02(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 2]),.din(mac_din),.acc_o(mac_acc_o[ 2]),.vld_o(mac_vld_o[ 2]));
mac u_mac_03(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 3]),.din(mac_din),.acc_o(mac_acc_o[ 3]),.vld_o(mac_vld_o[ 3]));
mac u_mac_04(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 4]),.din(mac_din),.acc_o(mac_acc_o[ 4]),.vld_o(mac_vld_o[ 4]));
mac u_mac_05(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 5]),.din(mac_din),.acc_o(mac_acc_o[ 5]),.vld_o(mac_vld_o[ 5]));
mac u_mac_06(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 6]),.din(mac_din),.acc_o(mac_acc_o[ 6]),.vld_o(mac_vld_o[ 6]));
mac u_mac_07(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 7]),.din(mac_din),.acc_o(mac_acc_o[ 7]),.vld_o(mac_vld_o[ 7]));
mac u_mac_08(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 8]),.din(mac_din),.acc_o(mac_acc_o[ 8]),.vld_o(mac_vld_o[ 8]));
mac u_mac_09(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[ 9]),.din(mac_din),.acc_o(mac_acc_o[ 9]),.vld_o(mac_vld_o[ 9]));
mac u_mac_10(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[10]),.din(mac_din),.acc_o(mac_acc_o[10]),.vld_o(mac_vld_o[10]));
mac u_mac_11(.clk(clk),.rstn(rstn),.vld_i(mac_vld_i),.win(mac_win[11]),.din(mac_din),.acc_o(mac_acc_o[11]),.vld_o(mac_vld_o[11]));

//------------------------------------------------------------
// MAC 입력 생성
//------------------------------------------------------------
wire        is_fr = (ctrl_row==0);
wire        is_lr = (ctrl_row==cur_height-1);
wire        is_fc = (ctrl_col==0);
wire        is_lc = (ctrl_col==cur_width-1);
wire [31:0] choff = ni_cnt * cur_width * cur_height;

integer k;
always @(*) begin
    mac_vld_i=0; mac_din=128'd0;
    for(k=0;k<TO;k=k+1) mac_win[k]=128'd0;
    if(ctrl_data_run && seq_state==SEQ_CONV) begin
        mac_vld_i=1;
        // 3x3 neighborhood packing (zero-padding at border)
        mac_din[ 7: 0]=(is_fr||is_fc)?8'd0:ifm_buf[choff+(ctrl_row-1)*cur_width+(ctrl_col-1)];
        mac_din[15: 8]=(is_fr       )?8'd0:ifm_buf[choff+(ctrl_row-1)*cur_width+ ctrl_col   ];
        mac_din[23:16]=(is_fr||is_lc)?8'd0:ifm_buf[choff+(ctrl_row-1)*cur_width+(ctrl_col+1)];
        mac_din[31:24]=(       is_fc)?8'd0:ifm_buf[choff+ ctrl_row   *cur_width+(ctrl_col-1)];
        mac_din[39:32]=               ifm_buf[choff+ ctrl_row   *cur_width+ ctrl_col         ];
        mac_din[47:40]=(       is_lc)?8'd0:ifm_buf[choff+ ctrl_row   *cur_width+(ctrl_col+1)];
        mac_din[55:48]=(is_lr||is_fc)?8'd0:ifm_buf[choff+(ctrl_row+1)*cur_width+(ctrl_col-1)];
        mac_din[63:56]=(is_lr       )?8'd0:ifm_buf[choff+(ctrl_row+1)*cur_width+ ctrl_col   ];
        mac_din[71:64]=(is_lr||is_lc)?8'd0:ifm_buf[choff+(ctrl_row+1)*cur_width+(ctrl_col+1)];

        // [수정1] 가중치 인덱스: f*filter_size + ni*9 + [0:8]
        // filter_size: CONV00=27, CONV02=144, CONV04=288
        for(k=0;k<TO;k=k+1) begin
            if((to_cnt*TO+k)<cur_no) begin
                case(layer_idx)
                2'd0: begin  // filter_size=27 (=9*3)
                    mac_win[k][ 7: 0] = w0[(to_cnt*TO+k)*27+ni_cnt*9+0];
                    mac_win[k][15: 8] = w0[(to_cnt*TO+k)*27+ni_cnt*9+1];
                    mac_win[k][23:16] = w0[(to_cnt*TO+k)*27+ni_cnt*9+2];
                    mac_win[k][31:24] = w0[(to_cnt*TO+k)*27+ni_cnt*9+3];
                    mac_win[k][39:32] = w0[(to_cnt*TO+k)*27+ni_cnt*9+4];
                    mac_win[k][47:40] = w0[(to_cnt*TO+k)*27+ni_cnt*9+5];
                    mac_win[k][55:48] = w0[(to_cnt*TO+k)*27+ni_cnt*9+6];
                    mac_win[k][63:56] = w0[(to_cnt*TO+k)*27+ni_cnt*9+7];
                    mac_win[k][71:64] = w0[(to_cnt*TO+k)*27+ni_cnt*9+8];
                end
                2'd1: begin  // filter_size=144 (=9*16)
                    mac_win[k][ 7: 0] = w1[(to_cnt*TO+k)*144+ni_cnt*9+0];
                    mac_win[k][15: 8] = w1[(to_cnt*TO+k)*144+ni_cnt*9+1];
                    mac_win[k][23:16] = w1[(to_cnt*TO+k)*144+ni_cnt*9+2];
                    mac_win[k][31:24] = w1[(to_cnt*TO+k)*144+ni_cnt*9+3];
                    mac_win[k][39:32] = w1[(to_cnt*TO+k)*144+ni_cnt*9+4];
                    mac_win[k][47:40] = w1[(to_cnt*TO+k)*144+ni_cnt*9+5];
                    mac_win[k][55:48] = w1[(to_cnt*TO+k)*144+ni_cnt*9+6];
                    mac_win[k][63:56] = w1[(to_cnt*TO+k)*144+ni_cnt*9+7];
                    mac_win[k][71:64] = w1[(to_cnt*TO+k)*144+ni_cnt*9+8];
                end
                default: begin  // filter_size=288 (=9*32)
                    mac_win[k][ 7: 0] = w2[(to_cnt*TO+k)*288+ni_cnt*9+0];
                    mac_win[k][15: 8] = w2[(to_cnt*TO+k)*288+ni_cnt*9+1];
                    mac_win[k][23:16] = w2[(to_cnt*TO+k)*288+ni_cnt*9+2];
                    mac_win[k][31:24] = w2[(to_cnt*TO+k)*288+ni_cnt*9+3];
                    mac_win[k][39:32] = w2[(to_cnt*TO+k)*288+ni_cnt*9+4];
                    mac_win[k][47:40] = w2[(to_cnt*TO+k)*288+ni_cnt*9+5];
                    mac_win[k][55:48] = w2[(to_cnt*TO+k)*288+ni_cnt*9+6];
                    mac_win[k][63:56] = w2[(to_cnt*TO+k)*288+ni_cnt*9+7];
                    mac_win[k][71:64] = w2[(to_cnt*TO+k)*288+ni_cnt*9+8];
                end
                endcase
            end
        end
    end
end

//------------------------------------------------------------
// MAC 출력 누산
//------------------------------------------------------------
always @(posedge clk, negedge rstn) begin
    if(!rstn) vld_pix_idx<=0;
    else begin
        if(seq_state==SEQ_CONV_START) vld_pix_idx<=0;
        else if(mac_vld_o[0] && vld_pix_idx<cur_width*cur_height) begin
            for(k=0;k<TO;k=k+1) begin
                if(ni_cnt==0)
                    accum[k][vld_pix_idx] <= $signed({{12{mac_acc_o[k][19]}},mac_acc_o[k]});
                else
                    accum[k][vld_pix_idx] <= accum[k][vld_pix_idx]
                                           + $signed({{12{mac_acc_o[k][19]}},mac_acc_o[k]});
            end
            vld_pix_idx<=vld_pix_idx+1;
        end
    end
end

//------------------------------------------------------------
// 메인 FSM
//------------------------------------------------------------
always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        seq_state<=SEQ_IDLE; layer_idx<=0; to_cnt<=0; ni_cnt<=0;
        cur_width<=256; cur_height<=256; cur_ni<=3; cur_no<=16; cur_shift<=7;
        ctrl_q_start<=0; ctrl_q_width<=0; ctrl_q_height<=0;
        ctrl_q_vsync_delay<=4; ctrl_q_hsync_delay<=4; ctrl_q_frame_size<=0;
        relu_idx<=0; pool_row<=0; pool_col<=0; pool_ch<=0;
        copy_idx<=0; copy_total<=0; drain_cnt<=0;
    end else begin
        ctrl_q_start<=0;
        case(seq_state)

        SEQ_IDLE: begin
            if(ap_start) begin
                layer_idx<=0; to_cnt<=0; ni_cnt<=0;
                cur_width<=f_width(0); cur_height<=f_height(0);
                cur_ni<=f_ni(0); cur_no<=f_no(0); cur_shift<=f_shift(0);
                seq_state<=SEQ_CONV_START;
                $display("[V4] CONV00 시작 (256x256, Ni=3, No=16, shift>>7)");
            end
        end

        SEQ_CONV_START: begin
            ctrl_q_width<=cur_width; ctrl_q_height<=cur_height;
            ctrl_q_vsync_delay<=4; ctrl_q_hsync_delay<=4;
            ctrl_q_frame_size<=cur_width*cur_height;
            ctrl_q_start<=1;
            seq_state<=SEQ_CONV;
        end

        SEQ_CONV: begin
            if(ctrl_end_frame) begin drain_cnt<=0; seq_state<=SEQ_DRAIN; end
        end

        SEQ_DRAIN: begin
            // MAC 파이프라인 비우기 (MAC_LATENCY=9 클럭)
            if(drain_cnt<MAC_LATENCY-1) drain_cnt<=drain_cnt+1;
            else begin
                drain_cnt<=0;
                if(ni_cnt<cur_ni-1) begin
                    ni_cnt<=ni_cnt+1; seq_state<=SEQ_CONV_START;
                end else begin
                    ni_cnt<=0; relu_idx<=0; seq_state<=SEQ_RELU;
                end
            end
        end

        // ① acc + bias  ② ReLU  ③ >>shift  ④ 클리핑[0,127]  ⑤ ofm_buf 저장
        SEQ_RELU: begin
            if(relu_idx<cur_width*cur_height) begin
                for(k=0;k<TO;k=k+1) begin
                    if((to_cnt*TO+k)<cur_no) begin
                        // ① 바이어스 덧셈
                        case(layer_idx)
                        2'd0: acc_bias = accum[k][relu_idx] + b0[to_cnt*TO+k];
                        2'd1: acc_bias = accum[k][relu_idx] + b1[to_cnt*TO+k];
                        default: acc_bias = accum[k][relu_idx] + b2[to_cnt*TO+k];
                        endcase
                        // ② ReLU + ③ >>shift + ④ 클리핑
                        if(acc_bias>0)
                            relu_out_r = (acc_bias>>cur_shift)>127 ? 8'd127
                                        : (acc_bias>>cur_shift);
                        else
                            relu_out_r = 8'd0;
                        // ⑤ 저장
                        ofm_buf[(to_cnt*TO+k)*cur_width*cur_height+relu_idx] <= relu_out_r;
                    end
                end
                relu_idx<=relu_idx+1;
            end else begin
                if((to_cnt+1)*TO<cur_no) begin
                    to_cnt<=to_cnt+1; ni_cnt<=0; relu_idx<=0;
                    seq_state<=SEQ_CONV_START;
                end else begin
                    to_cnt<=0; pool_row<=0; pool_col<=0; pool_ch<=0;
                    seq_state<=SEQ_MAXPOOL;
                    $display("[V4] Layer%0d CONV+Bias+ReLU 완료 -> MaxPool", layer_idx);
                end
            end
        end

        // [수정3] MaxPool: 지역변수(reg) 방식으로 1클럭에 처리
        // ofm_buf는 reg 배열이므로 조합 읽기 가능
        SEQ_MAXPOOL: begin
            begin : mp_blk
                reg [7:0] v00, v01, v10, v11, vmax;
                v00 = ofm_buf[pool_ch*cur_width*cur_height +  pool_row   *cur_width + pool_col  ];
                v01 = ofm_buf[pool_ch*cur_width*cur_height +  pool_row   *cur_width + pool_col+1];
                v10 = ofm_buf[pool_ch*cur_width*cur_height + (pool_row+1)*cur_width + pool_col  ];
                v11 = ofm_buf[pool_ch*cur_width*cur_height + (pool_row+1)*cur_width + pool_col+1];
                vmax = (v00>=v01)?((v00>=v10)?((v00>=v11)?v00:v11):((v10>=v11)?v10:v11)):
                                  ((v01>=v10)?((v01>=v11)?v01:v11):((v10>=v11)?v10:v11));
                pool_buf[pool_ch*(cur_width/2)*(cur_height/2)
                         +(pool_row/2)*(cur_width/2)+(pool_col/2)] <= vmax;
            end
            // 카운터: 채널 우선 → 열 → 행 순서 (stride=2)
            if(pool_ch < cur_no-1) pool_ch <= pool_ch+1;
            else begin
                pool_ch <= 0;
                if(pool_col+2 < cur_width) pool_col <= pool_col+2;
                else begin
                    pool_col <= 0;
                    if(pool_row+2 < cur_height) pool_row <= pool_row+2;
                    else begin
                        pool_row<=0; pool_col<=0; pool_ch<=0;
                        copy_idx<=0;
                        copy_total<=(cur_width/2)*(cur_height/2)*cur_no;
                        seq_state<=SEQ_COPY;
                        $display("[V4] Layer%0d MaxPool 완료", layer_idx);
                    end
                end
            end
        end

        SEQ_COPY: begin
            if(copy_idx<copy_total) begin
                ifm_buf[copy_idx]<=pool_buf[copy_idx];
                copy_idx<=copy_idx+1;
            end else seq_state<=SEQ_NEXT;
        end

        SEQ_NEXT: begin
            if(layer_idx<NUM_LAYERS-1) begin
                layer_idx<=layer_idx+1; to_cnt<=0; ni_cnt<=0; relu_idx<=0;
                pool_row<=0; pool_col<=0; pool_ch<=0;
                cur_width<=f_width(layer_idx+1); cur_height<=f_height(layer_idx+1);
                cur_ni<=f_ni(layer_idx+1); cur_no<=f_no(layer_idx+1);
                cur_shift<=f_shift(layer_idx+1);
                seq_state<=SEQ_CONV_START;
                $display("[V4] Layer%0d 시작", layer_idx+1);
            end else begin
                seq_state<=SEQ_DONE;
                $display("[V4] 완료: CONV00->Pool->CONV02->Pool->CONV04->Pool");
            end
        end

        SEQ_DONE: begin end
        default: seq_state<=SEQ_IDLE;
        endcase
    end
end

//------------------------------------------------------------
// AXI DMA (원본 유지)
//------------------------------------------------------------
axi_dma_ctrl #(.BIT_TRANS(BIT_TRANS)) u_dma_ctrl(
    .clk(clk),.rstn(rstn),.i_start(i_ctrl_reg0[0]),
    .i_base_address_rd(dram_base_addr_rd),.i_base_address_wr(dram_base_addr_wr),
    .i_num_trans(num_trans),.i_max_req_blk_idx(max_req_blk_idx),
    .i_read_done(read_done),.o_ctrl_read(ctrl_read),.o_read_addr(read_addr),
    .i_indata_req_wr(indata_req_wr),.i_write_done(write_done),
    .o_ctrl_write(ctrl_write),.o_write_addr(write_addr),
    .o_write_data_cnt(write_data_cnt),.o_ctrl_write_done(ctrl_write_done));

axi_dma_rd #(.BITS_TRANS(BIT_TRANS),.OUT_BITS_TRANS(OUT_BITS_TRANS),
    .AXI_WIDTH_USER(1),.AXI_WIDTH_ID(4),
    .AXI_WIDTH_AD(AXI_WIDTH_AD),.AXI_WIDTH_DA(AXI_WIDTH_DA),.AXI_WIDTH_DS(AXI_WIDTH_DS))
u_dma_read(
    .M_ARVALID(M_ARVALID),.M_ARREADY(M_ARREADY),.M_ARADDR(M_ARADDR),.M_ARID(M_ARID),
    .M_ARLEN(M_ARLEN),.M_ARSIZE(M_ARSIZE),.M_ARBURST(M_ARBURST),.M_ARLOCK(M_ARLOCK),
    .M_ARCACHE(M_ARCACHE),.M_ARPROT(M_ARPROT),.M_ARQOS(M_ARQOS),.M_ARREGION(M_ARREGION),
    .M_ARUSER(M_ARUSER),.M_RVALID(M_RVALID),.M_RREADY(M_RREADY),.M_RDATA(M_RDATA),
    .M_RLAST(M_RLAST),.M_RID(M_RID),.M_RUSER(),.M_RRESP(M_RRESP),
    .start_dma(ctrl_read),.num_trans(num_trans),.start_addr(read_addr),
    .data_o(read_data),.data_vld_o(read_data_vld),.data_cnt_o(read_data_cnt),
    .done_o(read_done),.clk(clk),.rstn(rstn));

dpram_wrapper #(.DEPTH(BUFF_DEPTH),.AW(BUFF_ADDR_W),.DW(AXI_WIDTH_DA)) u_data_buffer(
    .clk(clk),.ena(read_data_vld),.addra(read_data_cnt),.wea(read_data_vld),
    .dia(read_data),.enb(1'd1),.addrb(write_data_cnt),.dob(write_data));

axi_dma_wr #(.BITS_TRANS(BIT_TRANS),.OUT_BITS_TRANS(BIT_TRANS),
    .AXI_WIDTH_USER(1),.AXI_WIDTH_ID(4),
    .AXI_WIDTH_AD(AXI_WIDTH_AD),.AXI_WIDTH_DA(AXI_WIDTH_DA),.AXI_WIDTH_DS(AXI_WIDTH_DS))
u_dma_write(
    .M_AWID(M_AWID),.M_AWADDR(M_AWADDR),.M_AWLEN(M_AWLEN),.M_AWSIZE(M_AWSIZE),
    .M_AWBURST(M_AWBURST),.M_AWLOCK(M_AWLOCK),.M_AWCACHE(M_AWCACHE),.M_AWPROT(M_AWPROT),
    .M_AWREGION(M_AWREGION),.M_AWQOS(M_AWQOS),.M_AWVALID(M_AWVALID),.M_AWREADY(M_AWREADY),
    .M_AWUSER(),
    .M_WID(M_WID),.M_WDATA(M_WDATA),.M_WSTRB(M_WSTRB),.M_WLAST(M_WLAST),
    .M_WVALID(M_WVALID),.M_WREADY(M_WREADY),.M_WUSER(),.M_BUSER(),
    .M_BID(M_BID),.M_BRESP(M_BRESP),.M_BVALID(M_BVALID),.M_BREADY(M_BREADY),
    .start_dma(ctrl_write),.num_trans(num_trans),.start_addr(write_addr),
    .indata(write_data),.indata_req_o(indata_req_wr),.done_o(write_done),
    .fail_check(),.clk(clk),.rstn(rstn));

endmodule