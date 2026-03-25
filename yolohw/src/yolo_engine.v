//----------------------------------------------------------------+
// Project: Deep Learning Hardware Accelerator Design Contest
// Module:  yolo_engine (V3)
// Description:
//   CONV00 -> MaxPool -> CONV02 -> MaxPool -> CONV04 -> MaxPool
//
// 하드웨어 파라미터:
//   TO=12 (동시 출력채널), DSP 사용량 = 12×16 = 192개 (80%)
//   mac.v 1개 = 16개 곱셈기, 9클럭 파이프라인 지연
//
// 레이어별 처리:
//   CONV00: 256×256, Ni=3,  No=16, to_tiles=2, ni_clk=3
//   CONV02: 128×128, Ni=16, No=32, to_tiles=3, ni_clk=16
//   CONV04:  64×64,  Ni=32, No=64, to_tiles=6, ni_clk=32
//
// 시뮬레이션 전용:
//   가중치/IFM을 $readmemh로 직접 로드 (파일명만 사용)
//   hex 파일을 Vivado sim_1에 Add Sources로 추가 필요
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
    // AXI Read
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
    // AXI Write
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
    // 상태 출력
    output                       network_done,
    output                       network_done_led,
    output [AXI_WIDTH_DA-1:0]    read_data,
    output                       read_data_vld
);

`include "user_define_h.v"
`include "user_param_h.v"

// ============================================================
// 하드웨어 파라미터
// ============================================================
localparam TO          = 12;   // 동시 처리 출력채널 수 (mac 인스턴스 수)
localparam MAC_LATENCY = 9;    // mac.v 파이프라인 지연 (mul 5클럭 + adder_tree 4클럭)
localparam NUM_LAYERS  = 3;    // CONV00, CONV02, CONV04

// ============================================================
// 레이어 파라미터 함수 (Verilog-2001 호환, localparam 배열 대체)
// ============================================================
function [11:0] f_width;
    input [1:0] idx;
    case(idx) 2'd0: f_width=256; 2'd1: f_width=128; default: f_width=64; endcase
endfunction

function [11:0] f_height;
    input [1:0] idx;
    case(idx) 2'd0: f_height=256; 2'd1: f_height=128; default: f_height=64; endcase
endfunction

function [5:0] f_ni;
    input [1:0] idx;
    case(idx) 2'd0: f_ni=3; 2'd1: f_ni=16; default: f_ni=32; endcase
endfunction

function [6:0] f_no;
    input [1:0] idx;
    case(idx) 2'd0: f_no=16; 2'd1: f_no=32; default: f_no=64; endcase
endfunction

// ============================================================
// 가중치 버퍼 (32비트 워드, [7:0]만 사용)
// hex 파일을 Vivado sim_1에 Add Sources로 추가 후 파일명만 사용
// CONV00: 3×3×3×16  = 432
// CONV02: 3×3×16×32 = 4608
// CONV04: 3×3×32×64 = 18432
// ============================================================
reg [31:0] w0 [0:431  ];
reg [31:0] w1 [0:4607 ];
reg [31:0] w2 [0:18431];

// ============================================================
// IFM / OFM / Pool 버퍼 (8비트, 채널 우선(planar) 배치)
// 인덱스: buf[ch * W * H + row * W + col]
//
// ifm_buf: 최대 CONV00 입력 256×256×3 = 196608
//          이후 레이어는 pool 결과가 더 작으므로 재사용
// ofm_buf: 최대 CONV00 출력 256×256×16 = 1048576
// pool_buf: 최대 MaxPool 후 128×128×16 = 262144
// ============================================================
reg [7:0] ifm_buf  [0:196607 ];
reg [7:0] ofm_buf  [0:1048575];
reg [7:0] pool_buf [0:262143 ];

// ============================================================
// 누산기 (32비트 부호있는)
// accum[k][pixel_idx]: TO채널 × 최대 픽셀수(256×256=65536)
// ni_cnt==0 → 초기화(덮어쓰기), ni_cnt>0 → 누산
// ============================================================
reg signed [31:0] accum [0:TO-1][0:65535];

// ============================================================
// MAC 신호
// ============================================================
reg         mac_vld_i;
reg [127:0] mac_win [0:TO-1];
reg [127:0] mac_din;
wire [19:0] mac_acc_o [0:TO-1];
wire        mac_vld_o [0:TO-1];

// ============================================================
// cnn_ctrl 신호
// ============================================================
reg  [11:0] ctrl_q_width;
reg  [11:0] ctrl_q_height;
reg  [11:0] ctrl_q_vsync_delay;
reg  [11:0] ctrl_q_hsync_delay;
reg  [24:0] ctrl_q_frame_size;
reg         ctrl_q_start;
wire        ctrl_data_run;
wire [11:0] ctrl_row;
wire [11:0] ctrl_col;
wire        ctrl_end_frame;

// ============================================================
// FSM 상태 정의
// ============================================================
localparam SEQ_IDLE       = 4'd0;  // 시작 대기
localparam SEQ_CONV_START = 4'd1;  // cnn_ctrl q_start 1클럭 펄스
localparam SEQ_CONV       = 4'd2;  // MAC 연산 (ctrl_end_frame 대기)
localparam SEQ_DRAIN      = 4'd3;  // MAC 파이프라인 드레인 9클럭
localparam SEQ_RELU       = 4'd4;  // ReLU + >>12비트 → ofm_buf
localparam SEQ_MAXPOOL    = 4'd5;  // 2×2 MaxPool → pool_buf
localparam SEQ_COPY       = 4'd6;  // pool_buf → ifm_buf (다음 레이어 입력)
localparam SEQ_NEXT       = 4'd7;  // 다음 레이어 파라미터 갱신
localparam SEQ_DONE       = 4'd8;  // 완료

reg [3:0] seq_state;

// FSM 제어 변수
reg [1:0]  layer_idx;   // 0=CONV00, 1=CONV02, 2=CONV04
reg [3:0]  to_cnt;      // 출력채널 타일 인덱스 (0 ~ ceil(No/TO)-1)
reg [5:0]  ni_cnt;      // 입력채널 인덱스 (0 ~ Ni-1)

// 현재 레이어 파라미터 (등록값)
reg [11:0] cur_width;
reg [11:0] cur_height;
reg [5:0]  cur_ni;
reg [6:0]  cur_no;

// MAC 출력 픽셀 인덱스 (vld_o 기준)
reg [31:0] vld_pix_idx;

// ReLU 반복 인덱스
reg [31:0] relu_idx;

// MaxPool 위치
reg [11:0] pool_row;
reg [11:0] pool_col;
reg [6:0]  pool_ch;

// MaxPool 4픽셀 임시값 (모듈 레벨 선언 - always 내부 reg 불가)
reg [7:0] p_v00, p_v01, p_v10, p_v11, p_vmax;
// MaxPool pool_buf 저장을 위한 1클럭 지연 레지스터
reg [11:0] pool_row_d, pool_col_d;
reg [6:0]  pool_ch_d;
reg        pool_wr_en;

// SEQ_COPY 카운터
reg [31:0] copy_idx;
reg [31:0] copy_total;

// SEQ_DRAIN 카운터
reg [3:0]  drain_cnt;

// ============================================================
// 기존 AXI DMA 관련 신호 (원본 유지)
// ============================================================
localparam BIT_TRANS = BUFF_ADDR_W;

reg  ap_start;
reg  ap_done;
reg  interrupt;

reg  [31:0] dram_base_addr_rd;
reg  [31:0] dram_base_addr_wr;
reg  [31:0] reserved_register;

wire        ctrl_read;
wire        read_done;
wire [AXI_WIDTH_AD-1:0] read_addr;
wire [AXI_WIDTH_DA-1:0] read_data;
wire                     read_data_vld;
wire [BIT_TRANS-1:0]    read_data_cnt;

wire        ctrl_write_done;
wire        ctrl_write;
wire        write_done;
wire        indata_req_wr;
wire [BIT_TRANS-1:0]    write_data_cnt;
wire [AXI_WIDTH_AD-1:0] write_addr;
wire [AXI_WIDTH_DA-1:0] write_data;

wire [BIT_TRANS-1:0] num_trans      = 16;
wire [15:0] max_req_blk_idx         = (256*256) / 16;

// ============================================================
// 초기화: $readmemh (시뮬레이션 전용)
// hex 파일은 Vivado sim_1 (Simulation Sources)에 Add Sources 필요
// 파일명만 사용 → 경로 독립적
// ============================================================
integer init_i;
initial begin : LOAD_HEX
    reg [31:0] tmp_ifm [0:65535];

    // 가중치 로드 (파일명만 사용 - sim_1에 추가된 파일)
    $readmemh("CONV00_param_weight.hex", w0);
    $readmemh("CONV02_param_weight.hex", w1);
    $readmemh("CONV04_param_weight.hex", w2);

    // IFM 로드 (CONV00 입력: 256×256, 32비트 포맷)
    // 32비트 워드: [7:0]=R, [15:8]=G, [23:16]=B, [31:24]=0
    // ifm_buf 배치: 채널 우선(planar)
    //   ch0(R): ifm_buf[0 .. 65535]
    //   ch1(G): ifm_buf[65536 .. 131071]
    //   ch2(B): ifm_buf[131072 .. 196607]
    $readmemh("CONV00_input_32b.hex", tmp_ifm);
    for(init_i = 0; init_i < 65536; init_i = init_i + 1) begin
        ifm_buf[0*65536 + init_i] = tmp_ifm[init_i][7:0];    // R
        ifm_buf[1*65536 + init_i] = tmp_ifm[init_i][15:8];   // G
        ifm_buf[2*65536 + init_i] = tmp_ifm[init_i][23:16];  // B
    end

    $display("[yolo_engine] 가중치 및 IFM 로드 완료");
    $display("[yolo_engine] CONV00 입력 샘플: R[0]=%0d G[0]=%0d B[0]=%0d",
             ifm_buf[0], ifm_buf[65536], ifm_buf[131072]);
end

// ============================================================
// 제어 신호
// ============================================================
assign network_done     = interrupt;
assign network_done_led = interrupt;

always @(*) begin
    ap_done = ctrl_write_done;
end

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
        else if(seq_state == SEQ_DONE) interrupt <= 1;
    end
end

always @(posedge clk, negedge rstn) begin
    if(~rstn) begin
        dram_base_addr_rd <= 0;
        dram_base_addr_wr <= 0;
        reserved_register <= 0;
    end else begin
        if(!ap_start && i_ctrl_reg0[0]) begin
            dram_base_addr_rd <= i_ctrl_reg1;
            dram_base_addr_wr <= i_ctrl_reg2;
            reserved_register <= i_ctrl_reg3;
        end else if(ap_done) begin
            dram_base_addr_rd <= 0;
            dram_base_addr_wr <= 0;
            reserved_register <= 0;
        end
    end
end

// ============================================================
// cnn_ctrl 인스턴스
// ============================================================
cnn_ctrl u_cnn_ctrl (
    .clk             (clk                ),
    .rstn            (rstn               ),
    .q_width         (ctrl_q_width       ),
    .q_height        (ctrl_q_height      ),
    .q_vsync_delay   (ctrl_q_vsync_delay ),
    .q_hsync_delay   (ctrl_q_hsync_delay ),
    .q_frame_size    (ctrl_q_frame_size  ),
    .q_start         (ctrl_q_start       ),
    .o_ctrl_vsync_run(                   ),
    .o_ctrl_vsync_cnt(                   ),
    .o_ctrl_hsync_run(                   ),
    .o_ctrl_hsync_cnt(                   ),
    .o_ctrl_data_run (ctrl_data_run      ),
    .o_row           (ctrl_row           ),
    .o_col           (ctrl_col           ),
    .o_data_count    (                   ),
    .o_end_frame     (ctrl_end_frame     )
);

// ============================================================
// MAC 인스턴스 (TO=12, DSP 192개 = 80%)
// ============================================================
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

// ============================================================
// MAC 입력 생성 (조합회로)
// din: 현재 픽셀 기준 3×3 이웃을 128비트에 패킹 (zero padding)
// win[k]: k번째 mac의 필터 가중치
// ============================================================
wire        is_first_row = (ctrl_row == 0);
wire        is_last_row  = (ctrl_row == cur_height - 1);
wire        is_first_col = (ctrl_col == 0);
wire        is_last_col  = (ctrl_col == cur_width  - 1);
wire [31:0] ch_off       = ni_cnt * cur_width * cur_height;

integer k;
always @(*) begin
    mac_vld_i = 1'b0;
    mac_din   = 128'd0;
    for(k = 0; k < TO; k = k+1) mac_win[k] = 128'd0;

    if(ctrl_data_run && seq_state == SEQ_CONV) begin
        mac_vld_i = 1'b1;

        // 3×3 이웃 픽셀 패킹 (경계: zero padding)
        mac_din[ 7: 0] = (is_first_row||is_first_col) ? 8'd0 : ifm_buf[ch_off+(ctrl_row-1)*cur_width+(ctrl_col-1)];
        mac_din[15: 8] = (is_first_row              ) ? 8'd0 : ifm_buf[ch_off+(ctrl_row-1)*cur_width+ ctrl_col   ];
        mac_din[23:16] = (is_first_row|| is_last_col) ? 8'd0 : ifm_buf[ch_off+(ctrl_row-1)*cur_width+(ctrl_col+1)];
        mac_din[31:24] = (               is_first_col) ? 8'd0 : ifm_buf[ch_off+ ctrl_row   *cur_width+(ctrl_col-1)];
        mac_din[39:32] =                                         ifm_buf[ch_off+ ctrl_row   *cur_width+ ctrl_col   ];
        mac_din[47:40] = (                is_last_col) ? 8'd0 : ifm_buf[ch_off+ ctrl_row   *cur_width+(ctrl_col+1)];
        mac_din[55:48] = (is_last_row || is_first_col) ? 8'd0 : ifm_buf[ch_off+(ctrl_row+1)*cur_width+(ctrl_col-1)];
        mac_din[63:56] = (is_last_row               ) ? 8'd0 : ifm_buf[ch_off+(ctrl_row+1)*cur_width+ ctrl_col   ];
        mac_din[71:64] = (is_last_row ||  is_last_col) ? 8'd0 : ifm_buf[ch_off+(ctrl_row+1)*cur_width+(ctrl_col+1)];

        // TO개 출력채널 필터 패킹
        // 가중치 인덱스: (출력채널_절대) × 9 × Ni + ni_cnt × 9 + [0:8]
        for(k = 0; k < TO; k = k+1) begin
            if((to_cnt*TO + k) < cur_no) begin
                case(layer_idx)
                2'd0: begin
                    mac_win[k][ 7: 0] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+0][7:0];
                    mac_win[k][15: 8] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+1][7:0];
                    mac_win[k][23:16] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+2][7:0];
                    mac_win[k][31:24] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+3][7:0];
                    mac_win[k][39:32] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+4][7:0];
                    mac_win[k][47:40] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+5][7:0];
                    mac_win[k][55:48] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+6][7:0];
                    mac_win[k][63:56] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+7][7:0];
                    mac_win[k][71:64] = w0[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+8][7:0];
                end
                2'd1: begin
                    mac_win[k][ 7: 0] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+0][7:0];
                    mac_win[k][15: 8] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+1][7:0];
                    mac_win[k][23:16] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+2][7:0];
                    mac_win[k][31:24] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+3][7:0];
                    mac_win[k][39:32] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+4][7:0];
                    mac_win[k][47:40] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+5][7:0];
                    mac_win[k][55:48] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+6][7:0];
                    mac_win[k][63:56] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+7][7:0];
                    mac_win[k][71:64] = w1[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+8][7:0];
                end
                default: begin
                    mac_win[k][ 7: 0] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+0][7:0];
                    mac_win[k][15: 8] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+1][7:0];
                    mac_win[k][23:16] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+2][7:0];
                    mac_win[k][31:24] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+3][7:0];
                    mac_win[k][39:32] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+4][7:0];
                    mac_win[k][47:40] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+5][7:0];
                    mac_win[k][55:48] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+6][7:0];
                    mac_win[k][63:56] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+7][7:0];
                    mac_win[k][71:64] = w2[((to_cnt*TO+k)*9*cur_ni)+ni_cnt*9+8][7:0];
                end
                endcase
            end
        end
    end
end

// ============================================================
// MAC 출력 누산 (클럭 동기)
// vld_o=1이면 accum에 누산
// ni_cnt==0이면 덮어쓰기 (새 픽셀 시작), ni_cnt>0이면 누산
// ============================================================
always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        vld_pix_idx <= 0;
    end else begin
        // 새 프레임 시작 시 픽셀 인덱스 리셋
        if(seq_state == SEQ_CONV_START) begin
            vld_pix_idx <= 0;
        end else if(mac_vld_o[0] &&
                    vld_pix_idx < cur_width * cur_height) begin
            for(k = 0; k < TO; k = k+1) begin
                if(ni_cnt == 0)
                    // 첫 번째 입력채널: 초기화
                    accum[k][vld_pix_idx] <=
                        $signed({{12{mac_acc_o[k][19]}}, mac_acc_o[k]});
                else
                    // 이후 입력채널: 누산
                    accum[k][vld_pix_idx] <= accum[k][vld_pix_idx] +
                        $signed({{12{mac_acc_o[k][19]}}, mac_acc_o[k]});
            end
            vld_pix_idx <= vld_pix_idx + 1;
        end
    end
end

// ============================================================
// 메인 FSM
// ============================================================
always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        seq_state          <= SEQ_IDLE;
        layer_idx          <= 0;
        to_cnt             <= 0;
        ni_cnt             <= 0;
        cur_width          <= 256;
        cur_height         <= 256;
        cur_ni             <= 3;
        cur_no             <= 16;
        ctrl_q_start       <= 0;
        ctrl_q_width       <= 0;
        ctrl_q_height      <= 0;
        ctrl_q_vsync_delay <= 4;
        ctrl_q_hsync_delay <= 4;
        ctrl_q_frame_size  <= 0;
        relu_idx           <= 0;
        pool_row           <= 0;
        pool_col           <= 0;
        pool_ch            <= 0;
        copy_idx           <= 0;
        copy_total         <= 0;
        drain_cnt          <= 0;
    end else begin
        ctrl_q_start <= 0; // 기본값: 0 (필요 시 1클럭만 1)

        case(seq_state)

        // ------------------------------------------------
        // IDLE: ap_start 감지 → 첫 레이어 초기화
        // ------------------------------------------------
        SEQ_IDLE: begin
            if(ap_start) begin
                layer_idx  <= 0;
                to_cnt     <= 0;
                ni_cnt     <= 0;
                cur_width  <= f_width(0);    // 256
                cur_height <= f_height(0);   // 256
                cur_ni     <= f_ni(0);       // 3
                cur_no     <= f_no(0);       // 16
                seq_state  <= SEQ_CONV_START;
                $display("[V3] CONV00 시작 (256x256, Ni=3, No=16)");
            end
        end

        // ------------------------------------------------
        // CONV_START: cnn_ctrl 파라미터 세팅 + q_start 1클럭 펄스
        // ------------------------------------------------
        SEQ_CONV_START: begin
            ctrl_q_width       <= cur_width;
            ctrl_q_height      <= cur_height;
            ctrl_q_vsync_delay <= 4;
            ctrl_q_hsync_delay <= 4;
            ctrl_q_frame_size  <= cur_width * cur_height;
            ctrl_q_start       <= 1;  // 딱 1클럭만
            seq_state          <= SEQ_CONV;
        end

        // ------------------------------------------------
        // CONV: ctrl_end_frame까지 대기
        // MAC 누산은 별도 always 블록에서 처리됨
        // ------------------------------------------------
        SEQ_CONV: begin
            if(ctrl_end_frame) begin
                drain_cnt <= 0;
                seq_state <= SEQ_DRAIN;
            end
        end

        // ------------------------------------------------
        // DRAIN: MAC 파이프라인 9클럭 드레인
        // 마지막 픽셀들의 vld_o 출력을 기다림
        // ------------------------------------------------
        SEQ_DRAIN: begin
            if(drain_cnt < MAC_LATENCY - 1) begin
                drain_cnt <= drain_cnt + 1;
            end else begin
                drain_cnt <= 0;
                // ni_cnt 루프: 모든 입력채널 처리 완료?
                if(ni_cnt < cur_ni - 1) begin
                    ni_cnt    <= ni_cnt + 1;
                    seq_state <= SEQ_CONV_START; // 다음 입력채널
                end else begin
                    // 모든 입력채널 완료 → ReLU
                    ni_cnt    <= 0;
                    relu_idx  <= 0;
                    seq_state <= SEQ_RELU;
                    $display("[V3] Layer%0d to_cnt=%0d/%0d ReLU 시작",
                             layer_idx, to_cnt, (cur_no-1)/TO);
                end
            end
        end

        // ------------------------------------------------
        // RELU: ReLU + >>12비트 역양자화 → ofm_buf 저장
        // 1클럭에 TO개 픽셀(채널) 처리
        // ofm_buf[ch * W * H + pixel_idx]
        // ------------------------------------------------
        SEQ_RELU: begin
            if(relu_idx < cur_width * cur_height) begin
                for(k = 0; k < TO; k = k+1) begin
                    if((to_cnt*TO + k) < cur_no) begin
                        if(accum[k][relu_idx] > 0)
                            ofm_buf[(to_cnt*TO+k)*cur_width*cur_height + relu_idx]
                                <= accum[k][relu_idx][19:12]; // >>12 역양자화
                        else
                            ofm_buf[(to_cnt*TO+k)*cur_width*cur_height + relu_idx]
                                <= 8'd0; // ReLU
                    end
                end
                relu_idx <= relu_idx + 1;
            end else begin
                // to_cnt 타일 완료 → 다음 타일 or MaxPool
                if((to_cnt + 1) * TO < cur_no) begin
                    // 다음 출력채널 타일 처리
                    to_cnt    <= to_cnt + 1;
                    ni_cnt    <= 0;
                    relu_idx  <= 0;
                    seq_state <= SEQ_CONV_START;
                end else begin
                    // 모든 출력채널 완료 → MaxPool
                    to_cnt    <= 0;
                    pool_row  <= 0;
                    pool_col  <= 0;
                    pool_ch   <= 0;
                    seq_state <= SEQ_MAXPOOL;
                    $display("[V3] Layer%0d CONV 완료 → MaxPool 시작", layer_idx);
                end
            end
        end

        // ------------------------------------------------
        // MAXPOOL: 2×2 stride=2
        // 1클럭에 채널 1개의 위치 1개 처리
        // ofm_buf[ch*W*H + row*W + col] → pool_buf[ch*(W/2)*(H/2) + (row/2)*(W/2) + (col/2)]
        // ------------------------------------------------
        SEQ_MAXPOOL: begin
            // 4픽셀 읽기 (현재 클럭)
            p_v00 <= ofm_buf[pool_ch*cur_width*cur_height +  pool_row   *cur_width + pool_col  ];
            p_v01 <= ofm_buf[pool_ch*cur_width*cur_height +  pool_row   *cur_width + pool_col+1];
            p_v10 <= ofm_buf[pool_ch*cur_width*cur_height + (pool_row+1)*cur_width + pool_col  ];
            p_v11 <= ofm_buf[pool_ch*cur_width*cur_height + (pool_row+1)*cur_width + pool_col+1];

            // 최대값 계산 (다음 클럭에 p_vmax에 저장됨)
            p_vmax <= (p_v00 >= p_v01) ?
                       ((p_v00 >= p_v10) ? ((p_v00 >= p_v11) ? p_v00 : p_v11)
                                         : ((p_v10 >= p_v11) ? p_v10 : p_v11))
                     : ((p_v01 >= p_v10) ? ((p_v01 >= p_v11) ? p_v01 : p_v11)
                                         : ((p_v10 >= p_v11) ? p_v10 : p_v11));

            // 위치 전진 (channel → col → row 순서)
            if(pool_ch < cur_no - 1) begin
                pool_ch <= pool_ch + 1;
            end else begin
                pool_ch <= 0;
                if(pool_col + 2 < cur_width) begin
                    pool_col <= pool_col + 2;
                end else begin
                    pool_col <= 0;
                    if(pool_row + 2 < cur_height) begin
                        pool_row <= pool_row + 2;
                    end else begin
                        // MaxPool 완료 → COPY
                        pool_row   <= 0;
                        pool_col   <= 0;
                        pool_ch    <= 0;
                        copy_idx   <= 0;
                        copy_total <= (cur_width/2)*(cur_height/2)*cur_no;
                        seq_state  <= SEQ_COPY;
                        $display("[V3] Layer%0d MaxPool 완료", layer_idx);
                    end
                end
            end
        end

        // ------------------------------------------------
        // COPY: pool_buf → ifm_buf (1클럭 1바이트)
        // 다음 레이어의 입력으로 사용
        // ------------------------------------------------
        SEQ_COPY: begin
            if(copy_idx < copy_total) begin
                ifm_buf[copy_idx] <= pool_buf[copy_idx];
                copy_idx <= copy_idx + 1;
            end else begin
                seq_state <= SEQ_NEXT;
            end
        end

        // ------------------------------------------------
        // NEXT: 다음 레이어 파라미터 갱신
        // ------------------------------------------------
        SEQ_NEXT: begin
            if(layer_idx < NUM_LAYERS - 1) begin
                layer_idx  <= layer_idx + 1;
                to_cnt     <= 0;
                ni_cnt     <= 0;
                relu_idx   <= 0;
                pool_row   <= 0;
                pool_col   <= 0;
                pool_ch    <= 0;
                cur_width  <= f_width (layer_idx + 1);
                cur_height <= f_height(layer_idx + 1);
                cur_ni     <= f_ni    (layer_idx + 1);
                cur_no     <= f_no    (layer_idx + 1);
                seq_state  <= SEQ_CONV_START;
                $display("[V3] Layer%0d 시작", layer_idx + 1);
            end else begin
                seq_state <= SEQ_DONE;
                $display("[V3] 모든 레이어 완료!");
                $display("[V3] CONV00->MaxPool->CONV02->MaxPool->CONV04->MaxPool");
            end
        end

        // ------------------------------------------------
        // DONE: interrupt=1 (위 always 블록에서 처리)
        // ------------------------------------------------
        SEQ_DONE: begin
            // interrupt는 별도 always에서 seq_state==SEQ_DONE 감지
        end

        default: seq_state <= SEQ_IDLE;
        endcase
    end
end

// ============================================================
// MaxPool pool_buf 저장 (1클럭 지연 보정)
// SEQ_MAXPOOL에서 p_vmax는 1클럭 후에 유효하므로
// 별도 always로 지연된 위치 정보와 함께 pool_buf에 저장
// ============================================================
always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        pool_wr_en <= 0;
        pool_row_d <= 0;
        pool_col_d <= 0;
        pool_ch_d  <= 0;
    end else begin
        // 1클럭 지연
        pool_wr_en <= (seq_state == SEQ_MAXPOOL);
        pool_row_d <= pool_row;
        pool_col_d <= pool_col;
        pool_ch_d  <= pool_ch;

        // 지연된 p_vmax를 pool_buf에 저장
        if(pool_wr_en) begin
            pool_buf[pool_ch_d*(cur_width/2)*(cur_height/2)
                     + (pool_row_d/2)*(cur_width/2)
                     + (pool_col_d/2)] <= p_vmax;
        end
    end
end

// ============================================================
// 기존 AXI DMA 서브모듈 (원본 그대로 유지)
// ============================================================
axi_dma_ctrl #(.BIT_TRANS(BIT_TRANS))
u_dma_ctrl(
    .clk              (clk              ),
    .rstn             (rstn             ),
    .i_start          (i_ctrl_reg0[0]   ),
    .i_base_address_rd(dram_base_addr_rd),
    .i_base_address_wr(dram_base_addr_wr),
    .i_num_trans      (num_trans        ),
    .i_max_req_blk_idx(max_req_blk_idx  ),
    .i_read_done      (read_done        ),
    .o_ctrl_read      (ctrl_read        ),
    .o_read_addr      (read_addr        ),
    .i_indata_req_wr  (indata_req_wr    ),
    .i_write_done     (write_done       ),
    .o_ctrl_write     (ctrl_write       ),
    .o_write_addr     (write_addr       ),
    .o_write_data_cnt (write_data_cnt   ),
    .o_ctrl_write_done(ctrl_write_done  )
);

axi_dma_rd #(
    .BITS_TRANS    (BIT_TRANS      ),
    .OUT_BITS_TRANS(OUT_BITS_TRANS ),
    .AXI_WIDTH_USER(1              ),
    .AXI_WIDTH_ID  (4              ),
    .AXI_WIDTH_AD  (AXI_WIDTH_AD  ),
    .AXI_WIDTH_DA  (AXI_WIDTH_DA  ),
    .AXI_WIDTH_DS  (AXI_WIDTH_DS  ))
u_dma_read(
    .M_ARVALID (M_ARVALID ), .M_ARREADY (M_ARREADY ),
    .M_ARADDR  (M_ARADDR  ), .M_ARID    (M_ARID    ),
    .M_ARLEN   (M_ARLEN   ), .M_ARSIZE  (M_ARSIZE  ),
    .M_ARBURST (M_ARBURST ), .M_ARLOCK  (M_ARLOCK  ),
    .M_ARCACHE (M_ARCACHE ), .M_ARPROT  (M_ARPROT  ),
    .M_ARQOS   (M_ARQOS   ), .M_ARREGION(M_ARREGION),
    .M_ARUSER  (M_ARUSER  ), .M_RVALID  (M_RVALID  ),
    .M_RREADY  (M_RREADY  ), .M_RDATA   (M_RDATA   ),
    .M_RLAST   (M_RLAST   ), .M_RID     (M_RID     ),
    .M_RUSER   (          ), .M_RRESP   (M_RRESP   ),
    .start_dma (ctrl_read ), .num_trans (num_trans  ),
    .start_addr(read_addr ), .data_o    (read_data  ),
    .data_vld_o(read_data_vld), .data_cnt_o(read_data_cnt),
    .done_o    (read_done ), .clk       (clk        ),
    .rstn      (rstn      )
);

dpram_wrapper #(
    .DEPTH(BUFF_DEPTH  ),
    .AW   (BUFF_ADDR_W ),
    .DW   (AXI_WIDTH_DA))
u_data_buffer(
    .clk  (clk          ),
    .ena  (read_data_vld ),
    .addra(read_data_cnt ),
    .wea  (read_data_vld ),
    .dia  (read_data     ),
    .enb  (1'd1          ),
    .addrb(write_data_cnt),
    .dob  (write_data    )
);

axi_dma_wr #(
    .BITS_TRANS    (BIT_TRANS    ),
    .OUT_BITS_TRANS(BIT_TRANS    ),
    .AXI_WIDTH_USER(1            ),
    .AXI_WIDTH_ID  (4            ),
    .AXI_WIDTH_AD  (AXI_WIDTH_AD ),
    .AXI_WIDTH_DA  (AXI_WIDTH_DA ),
    .AXI_WIDTH_DS  (AXI_WIDTH_DS ))
u_dma_write(
    .M_AWID    (M_AWID    ), .M_AWADDR (M_AWADDR ),
    .M_AWLEN   (M_AWLEN   ), .M_AWSIZE (M_AWSIZE ),
    .M_AWBURST (M_AWBURST ), .M_AWLOCK (M_AWLOCK ),
    .M_AWCACHE (M_AWCACHE ), .M_AWPROT (M_AWPROT ),
    .M_AWREGION(M_AWREGION), .M_AWQOS  (M_AWQOS  ),
    .M_AWVALID (M_AWVALID ), .M_AWREADY(M_AWREADY),
    .M_AWUSER  (          ),
    .M_WID     (M_WID     ), .M_WDATA  (M_WDATA  ),
    .M_WSTRB   (M_WSTRB   ), .M_WLAST  (M_WLAST  ),
    .M_WVALID  (M_WVALID  ), .M_WREADY (M_WREADY ),
    .M_WUSER   (          ), .M_BUSER  (         ),
    .M_BID     (M_BID     ), .M_BRESP  (M_BRESP  ),
    .M_BVALID  (M_BVALID  ), .M_BREADY (M_BREADY ),
    .start_dma (ctrl_write), .num_trans(num_trans ),
    .start_addr(write_addr), .indata   (write_data),
    .indata_req_o(indata_req_wr), .done_o(write_done),
    .fail_check(          ), .clk      (clk       ),
    .rstn      (rstn      )
);

endmodule
