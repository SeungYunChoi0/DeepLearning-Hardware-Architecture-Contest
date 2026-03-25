//----------------------------------------------------------------+
// Project: Deep Learning Hardware Design Contest
// Module: yolo_engine (V2 - CONV + MaxPool 구현)
// Description:
//      CONV00 → MaxPool → CONV02 → MaxPool → CONV04 → MaxPool
//
// 수정사항:
//   - TO=12 MAC 인스턴스 추가
//   - cnn_ctrl 연결
//   - Ni채널 누산(accumulation) 로직
//   - ReLU + 역양자화 (>>12비트)
//   - MaxPool 2×2 stride=2
//   - 3레이어 시퀀서 FSM
//----------------------------------------------------------------+
module yolo_engine #(
    parameter AXI_WIDTH_AD = 32,
    parameter AXI_WIDTH_ID = 4,
    parameter AXI_WIDTH_DA = 32,
    parameter AXI_WIDTH_DS = AXI_WIDTH_DA/8,
    parameter OUT_BITS_TRANS = 18,
    parameter WBUF_AW = 9,
    parameter WBUF_DW = 8*3*3*16,
    parameter WBUF_DS = WBUF_DW/8,
    parameter MEM_BASE_ADDR = 'h8000_0000,
    parameter MEM_DATA_BASE_ADDR = 4096
)
(
      input                          clk
    , input                          rstn

    , input [31:0] i_ctrl_reg0
    , input [31:0] i_ctrl_reg1
    , input [31:0] i_ctrl_reg2
    , input [31:0] i_ctrl_reg3

    , output                         M_ARVALID
    , input                          M_ARREADY
    , output  [AXI_WIDTH_AD-1:0]     M_ARADDR
    , output  [AXI_WIDTH_ID-1:0]     M_ARID
    , output  [7:0]                  M_ARLEN
    , output  [2:0]                  M_ARSIZE
    , output  [1:0]                  M_ARBURST
    , output  [1:0]                  M_ARLOCK
    , output  [3:0]                  M_ARCACHE
    , output  [2:0]                  M_ARPROT
    , output  [3:0]                  M_ARQOS
    , output  [3:0]                  M_ARREGION
    , output  [3:0]                  M_ARUSER
    , input                          M_RVALID
    , output                         M_RREADY
    , input  [AXI_WIDTH_DA-1:0]      M_RDATA
    , input                          M_RLAST
    , input  [AXI_WIDTH_ID-1:0]      M_RID
    , input  [3:0]                   M_RUSER
    , input  [1:0]                   M_RRESP

    , output                         M_AWVALID
    , input                          M_AWREADY
    , output  [AXI_WIDTH_AD-1:0]     M_AWADDR
    , output  [AXI_WIDTH_ID-1:0]     M_AWID
    , output  [7:0]                  M_AWLEN
    , output  [2:0]                  M_AWSIZE
    , output  [1:0]                  M_AWBURST
    , output  [1:0]                  M_AWLOCK
    , output  [3:0]                  M_AWCACHE
    , output  [2:0]                  M_AWPROT
    , output  [3:0]                  M_AWQOS
    , output  [3:0]                  M_AWREGION
    , output  [3:0]                  M_AWUSER

    , output                         M_WVALID
    , input                          M_WREADY
    , output  [AXI_WIDTH_DA-1:0]     M_WDATA
    , output  [AXI_WIDTH_DS-1:0]     M_WSTRB
    , output                         M_WLAST
    , output  [AXI_WIDTH_ID-1:0]     M_WID
    , output  [3:0]                  M_WUSER

    , input                          M_BVALID
    , output                         M_BREADY
    , input  [1:0]                   M_BRESP
    , input  [AXI_WIDTH_ID-1:0]      M_BID
    , input                          M_BUSER

    , output network_done
    , output network_done_led
    , output [AXI_WIDTH_DA-1:0]      read_data
    , output                         read_data_vld
);

`include "user_define_h.v"
`include "user_param_h.v"

// ============================================================
// [V2] 하드웨어 파라미터
// ============================================================
localparam TO = 12;         // 출력채널 타일링 (동시 처리 출력채널 수)
localparam MAC_LATENCY = 9; // mac.v 파이프라인 지연 (5클럭 mul + 4클럭 adder_tree)

// ============================================================
// [V2] 레이어 파라미터 (3개 레이어: CONV00, CONV02, CONV04)
// ============================================================
// layer_idx: 0=CONV00, 1=CONV02, 2=CONV04
localparam NUM_LAYERS = 3;

// 각 레이어 IFM 크기
localparam [11:0] L_WIDTH  [0:2] = '{256, 128, 64};
localparam [11:0] L_HEIGHT [0:2] = '{256, 128, 64};
// 각 레이어 입출력 채널 수
localparam [7:0]  L_NI     [0:2] = '{3,  16, 32};
localparam [7:0]  L_NO     [0:2] = '{16, 32, 64};
// MaxPool 후 크기 (입력의 절반)
localparam [11:0] L_POOL_W [0:2] = '{128, 64, 32};
localparam [11:0] L_POOL_H [0:2] = '{128, 64, 32};

// ============================================================
// [V2] 가중치 버퍼 (시뮬레이션: $readmemh로 로드)
// 최대 크기: CONV04 = 3×3×32×64 = 18432
// ============================================================
reg [7:0] weight_buf_l0 [0:431   ];  // CONV00: 3×3×3×16  = 432
reg [7:0] weight_buf_l1 [0:4607  ];  // CONV02: 3×3×16×32 = 4608
reg [7:0] weight_buf_l2 [0:18431 ];  // CONV04: 3×3×32×64 = 18432

// ============================================================
// [V2] IFM/OFM 버퍼 (시뮬레이션용 reg 배열)
// CONV00 IFM: 256×256×3   = 196608
// CONV00 OFM: 256×256×16  = 1048576  (MaxPool 후: 128×128×16 = 262144)
// CONV02 IFM: 128×128×16  = 262144
// CONV02 OFM: 128×128×32  = 524288   (MaxPool 후:  64×64×32  = 131072)
// CONV04 IFM:  64×64×32   = 131072
// CONV04 OFM:  64×64×64   = 262144   (MaxPool 후:  32×32×64  =  65536)
// ============================================================
reg [7:0] ifm_buf  [0:196607];   // 입력 feature map (최대 256×256×3)
reg [7:0] ofm_buf  [0:1048575];  // 출력 feature map (최대 256×256×16)
reg [7:0] pool_buf [0:262143];   // MaxPool 출력 (최대 128×128×16)

// ============================================================
// [V2] 누산기 (Accumulator)
// MAC 출력 20비트를 채널별로 누산
// ============================================================
reg signed [31:0] accum [0:TO-1][0:65535]; // TO채널 × 최대 픽셀수(256×256)
integer acc_i;

// ============================================================
// [V2] MAC 신호
// ============================================================
reg         mac_vld_i;
reg  [127:0] mac_win [0:TO-1];
reg  [127:0] mac_din;
wire [19:0]  mac_acc_o [0:TO-1];
wire         mac_vld_o [0:TO-1];

// ============================================================
// [V2] cnn_ctrl 신호
// ============================================================
reg  [11:0] ctrl_q_width;
reg  [11:0] ctrl_q_height;
reg  [11:0] ctrl_q_vsync_delay;
reg  [11:0] ctrl_q_hsync_delay;
reg  [24:0] ctrl_q_frame_size;
reg         ctrl_q_start;
wire        ctrl_vsync_run;
wire [11:0] ctrl_vsync_cnt;
wire        ctrl_hsync_run;
wire [11:0] ctrl_hsync_cnt;
wire        ctrl_data_run;
wire [11:0] ctrl_row;
wire [11:0] ctrl_col;
wire [24:0] ctrl_data_count;
wire        ctrl_end_frame;

// ============================================================
// [V2] 레이어 시퀀서 FSM
// ============================================================
localparam SEQ_IDLE     = 4'd0;
localparam SEQ_LOAD_W   = 4'd1;  // 가중치 로드 (이미 $readmemh로 완료)
localparam SEQ_CONV     = 4'd2;  // CONV 연산
localparam SEQ_ACCUM    = 4'd3;  // 채널 누산
localparam SEQ_RELU     = 4'd4;  // ReLU + 역양자화
localparam SEQ_MAXPOOL  = 4'd5;  // MaxPool
localparam SEQ_NEXT     = 4'd6;  // 다음 레이어로
localparam SEQ_DONE     = 4'd7;  // 완료

reg [3:0]  seq_state;
reg [1:0]  layer_idx;   // 현재 레이어 인덱스 (0~2)
reg [7:0]  ni_cnt;      // 현재 처리 중인 입력채널 그룹
reg [7:0]  to_cnt;      // 현재 처리 중인 출력채널 타일
reg [11:0] cur_width;
reg [11:0] cur_height;
reg [7:0]  cur_ni;
reg [7:0]  cur_no;

// 픽셀 위치 (conv 루프)
reg [11:0] conv_row;
reg [11:0] conv_col;

// 처리 완료 플래그
reg conv_done;
reg pool_done;

// ============================================================
// 기존 AXI DMA 관련 신호 (원본 유지)
// ============================================================
localparam BIT_TRANS = BUFF_ADDR_W;

reg ap_start;
reg ap_ready;
reg ap_done;
reg interrupt;

reg [31:0] dram_base_addr_rd;
reg [31:0] dram_base_addr_wr;
reg [31:0] reserved_register;

wire ctrl_read;
wire read_done;
wire [AXI_WIDTH_AD-1:0] read_addr;
wire [AXI_WIDTH_DA-1:0] read_data;
wire                    read_data_vld;
wire [BIT_TRANS   -1:0] read_data_cnt;

wire ctrl_write_done;
wire ctrl_write;
wire write_done;
wire indata_req_wr;
wire [BIT_TRANS   -1:0] write_data_cnt;
wire [AXI_WIDTH_AD-1:0] write_addr;
wire [AXI_WIDTH_DA-1:0] write_data;

wire [BIT_TRANS-1:0] num_trans       = 16;
wire [15:0] max_req_blk_idx          = (256*256)/16;

// ============================================================
// [V2] 초기화 - 가중치 로드 (시뮬레이션)
// ============================================================
initial begin
    $readmemh("C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_param/CONV00_param_weight.hex", weight_buf_l0);
    $readmemh("C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_param/CONV02_param_weight.hex", weight_buf_l1);
    $readmemh("C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_param/CONV04_param_weight.hex", weight_buf_l2);
    $display("[V2] Weights loaded for CONV00, CONV02, CONV04");
end

// ============================================================
// 제어 신호
// ============================================================
always @ (*) begin
    ap_done  = ctrl_write_done;
    ap_ready = 1;
end
assign network_done     = interrupt;
assign network_done_led = interrupt;

always @ (posedge clk, negedge rstn) begin
    if(~rstn) ap_start <= 0;
    else begin
        if(!ap_start && i_ctrl_reg0[0]) ap_start <= 1;
        else if (ap_done)               ap_start <= 0;
    end
end

always @(posedge clk, negedge rstn) begin
    if(~rstn) interrupt <= 0;
    else begin
        if(i_ctrl_reg0[0])     interrupt <= 0;
        else if (seq_state == SEQ_DONE) interrupt <= 1;
    end
end

always @ (posedge clk, negedge rstn) begin
    if(~rstn) begin
        dram_base_addr_rd <= 0;
        dram_base_addr_wr <= 0;
        reserved_register <= 0;
    end else begin
        if(!ap_start && i_ctrl_reg0[0]) begin
            dram_base_addr_rd <= i_ctrl_reg1;
            dram_base_addr_wr <= i_ctrl_reg2;
            reserved_register <= i_ctrl_reg3;
        end else if (ap_done) begin
            dram_base_addr_rd <= 0;
            dram_base_addr_wr <= 0;
            reserved_register <= 0;
        end
    end
end

// ============================================================
// [V2] cnn_ctrl 인스턴스
// ============================================================
cnn_ctrl u_cnn_ctrl (
    .clk            (clk            ),
    .rstn           (rstn           ),
    .q_width        (ctrl_q_width   ),
    .q_height       (ctrl_q_height  ),
    .q_vsync_delay  (ctrl_q_vsync_delay),
    .q_hsync_delay  (ctrl_q_hsync_delay),
    .q_frame_size   (ctrl_q_frame_size ),
    .q_start        (ctrl_q_start   ),
    .o_ctrl_vsync_run(ctrl_vsync_run),
    .o_ctrl_vsync_cnt(ctrl_vsync_cnt),
    .o_ctrl_hsync_run(ctrl_hsync_run),
    .o_ctrl_hsync_cnt(ctrl_hsync_cnt),
    .o_ctrl_data_run (ctrl_data_run ),
    .o_row           (ctrl_row      ),
    .o_col           (ctrl_col      ),
    .o_data_count    (ctrl_data_count),
    .o_end_frame     (ctrl_end_frame)
);

// ============================================================
// [V2] MAC 인스턴스 (TO=12개)
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
// [V2] MAC 입력 생성 (din + win 패킹)
// ============================================================
// 경계 플래그
wire is_first_row = (ctrl_row == 0);
wire is_last_row  = (ctrl_row == cur_height - 1);
wire is_first_col = (ctrl_col == 0);
wire is_last_col  = (ctrl_col == cur_width - 1);

// 현재 입력채널 오프셋 (ifm_buf에서 ni_cnt번째 채널)
wire [31:0] ch_offset = ni_cnt * cur_width * cur_height;

integer k;
always @(*) begin
    mac_vld_i = 0;
    mac_din   = 128'd0;
    for(k = 0; k < TO; k = k+1)
        mac_win[k] = 128'd0;

    if(ctrl_data_run && seq_state == SEQ_CONV) begin
        mac_vld_i = 1;

        // 3×3 이웃 픽셀 패킹 (현재 입력채널 ni_cnt)
        mac_din[ 7: 0] = (is_first_row||is_first_col) ? 8'd0 : ifm_buf[ch_offset+(ctrl_row-1)*cur_width+(ctrl_col-1)];
        mac_din[15: 8] = (is_first_row              ) ? 8'd0 : ifm_buf[ch_offset+(ctrl_row-1)*cur_width+ ctrl_col   ];
        mac_din[23:16] = (is_first_row|| is_last_col) ? 8'd0 : ifm_buf[ch_offset+(ctrl_row-1)*cur_width+(ctrl_col+1)];
        mac_din[31:24] = (              is_first_col) ? 8'd0 : ifm_buf[ch_offset+ ctrl_row   *cur_width+(ctrl_col-1)];
        mac_din[39:32] =                                        ifm_buf[ch_offset+ ctrl_row   *cur_width+ ctrl_col   ];
        mac_din[47:40] = (               is_last_col) ? 8'd0 : ifm_buf[ch_offset+ ctrl_row   *cur_width+(ctrl_col+1)];
        mac_din[55:48] = (is_last_row || is_first_col)? 8'd0 : ifm_buf[ch_offset+(ctrl_row+1)*cur_width+(ctrl_col-1)];
        mac_din[63:56] = (is_last_row               ) ? 8'd0 : ifm_buf[ch_offset+(ctrl_row+1)*cur_width+ ctrl_col   ];
        mac_din[71:64] = (is_last_row ||  is_last_col)? 8'd0 : ifm_buf[ch_offset+(ctrl_row+1)*cur_width+(ctrl_col+1)];

        // TO개 출력채널의 필터 패킹 (to_cnt*TO + j 번째 출력채널)
        for(k = 0; k < TO; k = k+1) begin
            // 가중치 인덱스: (출력채널) × (Fx×Fy×Ni) + ni_cnt × (Fx×Fy) + [0:8]
            // layer별 weight_buf 선택
            if(layer_idx == 0) begin
                mac_win[k][ 7: 0] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 0];
                mac_win[k][15: 8] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 1];
                mac_win[k][23:16] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 2];
                mac_win[k][31:24] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 3];
                mac_win[k][39:32] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 4];
                mac_win[k][47:40] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 5];
                mac_win[k][55:48] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 6];
                mac_win[k][63:56] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 7];
                mac_win[k][71:64] = weight_buf_l0[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 8];
            end else if(layer_idx == 1) begin
                mac_win[k][ 7: 0] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 0];
                mac_win[k][15: 8] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 1];
                mac_win[k][23:16] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 2];
                mac_win[k][31:24] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 3];
                mac_win[k][39:32] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 4];
                mac_win[k][47:40] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 5];
                mac_win[k][55:48] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 6];
                mac_win[k][63:56] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 7];
                mac_win[k][71:64] = weight_buf_l1[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 8];
            end else begin
                mac_win[k][ 7: 0] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 0];
                mac_win[k][15: 8] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 1];
                mac_win[k][23:16] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 2];
                mac_win[k][31:24] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 3];
                mac_win[k][39:32] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 4];
                mac_win[k][47:40] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 5];
                mac_win[k][55:48] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 6];
                mac_win[k][63:56] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 7];
                mac_win[k][71:64] = weight_buf_l2[((to_cnt*TO+k)*3*3*cur_ni) + ni_cnt*9 + 8];
            end
        end
    end
end

// ============================================================
// [V2] MAC 출력 누산
// vld_o가 1이면 acc_o를 accum에 더함
// ============================================================
reg [31:0] vld_pix_idx; // vld_o가 1인 시점의 픽셀 인덱스
reg        vld_prev;

always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        vld_pix_idx <= 0;
        vld_prev    <= 0;
    end else begin
        vld_prev <= mac_vld_o[0];
        if(mac_vld_o[0]) begin
            // TO개 채널 동시 누산
            for(k = 0; k < TO; k = k+1) begin
                // ni_cnt==0이면 새로운 픽셀 시작 (초기화 후 누산)
                if(ni_cnt == 0)
                    accum[k][vld_pix_idx] <= $signed(mac_acc_o[k]);
                else
                    accum[k][vld_pix_idx] <= accum[k][vld_pix_idx] + $signed(mac_acc_o[k]);
            end
            // 픽셀 인덱스 증가
            if(vld_pix_idx == cur_width*cur_height - 1)
                vld_pix_idx <= 0;
            else
                vld_pix_idx <= vld_pix_idx + 1;
        end else if(ctrl_q_start) begin
            vld_pix_idx <= 0;
        end
    end
end

// ============================================================
// [V2] 레이어 시퀀서 FSM
// ============================================================
integer p, q;
reg [31:0] relu_idx;    // ReLU 처리 중인 픽셀 인덱스
reg [31:0] pool_row_r, pool_col_r; // MaxPool 위치

always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        seq_state       <= SEQ_IDLE;
        layer_idx       <= 0;
        ni_cnt          <= 0;
        to_cnt          <= 0;
        cur_width       <= 256;
        cur_height      <= 256;
        cur_ni          <= 3;
        cur_no          <= 16;
        conv_done       <= 0;
        pool_done       <= 0;
        ctrl_q_start    <= 0;
        ctrl_q_width    <= 0;
        ctrl_q_height   <= 0;
        ctrl_q_vsync_delay <= 9;
        ctrl_q_hsync_delay <= 9;
        ctrl_q_frame_size  <= 0;
        relu_idx        <= 0;
        pool_row_r      <= 0;
        pool_col_r      <= 0;
        // accum 초기화
        for(acc_i = 0; acc_i < TO; acc_i = acc_i+1)
            for(p = 0; p < 65536; p = p+1)
                accum[acc_i][p] <= 0;
    end else begin
        case(seq_state)
        // ----- IDLE: 시작 대기 -----
        SEQ_IDLE: begin
            if(ap_start) begin
                $display("[V2] Start processing - Layer 0 (CONV00)");
                layer_idx  <= 0;
                to_cnt     <= 0;
                ni_cnt     <= 0;
                cur_width  <= L_WIDTH[0];
                cur_height <= L_HEIGHT[0];
                cur_ni     <= L_NI[0];
                cur_no     <= L_NO[0];
                seq_state  <= SEQ_CONV;
            end
        end

        // ----- CONV: MAC 연산 (cnn_ctrl 사용) -----
        SEQ_CONV: begin
            // cnn_ctrl 설정 및 시작
            ctrl_q_width       <= cur_width;
            ctrl_q_height      <= cur_height;
            ctrl_q_vsync_delay <= 9;
            ctrl_q_hsync_delay <= 9;
            ctrl_q_frame_size  <= cur_width * cur_height;
            ctrl_q_start       <= 1;

            // cnn_ctrl이 한 프레임 완료되면
            if(ctrl_end_frame) begin
                ctrl_q_start <= 0;
                // 입력채널 다음으로
                if(ni_cnt < cur_ni - 1) begin
                    ni_cnt <= ni_cnt + 1;
                    // 같은 출력채널 타일에 대해 다음 입력채널 처리
                end else begin
                    // 모든 입력채널 처리 완료 → ReLU 단계로
                    ni_cnt    <= 0;
                    relu_idx  <= 0;
                    seq_state <= SEQ_RELU;
                end
            end
        end

        // ----- RELU: ReLU + 역양자화 + OFM 저장 -----
        SEQ_RELU: begin
            if(relu_idx < cur_width * cur_height) begin
                for(k = 0; k < TO; k = k+1) begin
                    // ofm_buf[(to_cnt*TO + k) * W * H + pixel_idx]
                    if(accum[k][relu_idx] > 0)
                        ofm_buf[(to_cnt*TO+k)*cur_width*cur_height + relu_idx]
                            <= accum[k][relu_idx][19:12]; // >>12 역양자화
                    else
                        ofm_buf[(to_cnt*TO+k)*cur_width*cur_height + relu_idx] <= 8'd0;
                end
                relu_idx <= relu_idx + 1;
            end else begin
                // 다음 TO 타일로
                if((to_cnt+1)*TO < cur_no) begin
                    to_cnt    <= to_cnt + 1;
                    relu_idx  <= 0;
                    seq_state <= SEQ_CONV; // 다음 출력채널 타일 처리
                end else begin
                    // 모든 출력채널 처리 완료 → MaxPool
                    to_cnt        <= 0;
                    pool_row_r    <= 0;
                    pool_col_r    <= 0;
                    seq_state     <= SEQ_MAXPOOL;
                    $display("[V2] Layer %0d CONV done, starting MaxPool", layer_idx);
                end
            end
        end

        // ----- MAXPOOL: 2×2 stride=2 -----
        SEQ_MAXPOOL: begin
            if(pool_row_r < cur_height && pool_col_r < cur_width) begin
                for(k = 0; k < cur_no; k = k+1) begin
                    // 4개 중 최대값 선택
                    // ofm_buf[ch * W * H + row * W + col]
                    reg [7:0] v00, v01, v10, v11, vmax;
                    v00 = ofm_buf[k*cur_width*cur_height + pool_row_r*cur_width     + pool_col_r    ];
                    v01 = ofm_buf[k*cur_width*cur_height + pool_row_r*cur_width     + pool_col_r + 1];
                    v10 = ofm_buf[k*cur_width*cur_height + (pool_row_r+1)*cur_width + pool_col_r    ];
                    v11 = ofm_buf[k*cur_width*cur_height + (pool_row_r+1)*cur_width + pool_col_r + 1];
                    vmax = (v00>v01)?((v00>v10)?((v00>v11)?v00:v11):((v10>v11)?v10:v11)):
                                     ((v01>v10)?((v01>v11)?v01:v11):((v10>v11)?v10:v11));
                    // pool_buf[ch * W/2 * H/2 + (row/2) * (W/2) + (col/2)]
                    pool_buf[k*(cur_width/2)*(cur_height/2) + (pool_row_r/2)*(cur_width/2) + (pool_col_r/2)] <= vmax;
                end
                // 2씩 증가 (stride=2)
                if(pool_col_r + 2 < cur_width)
                    pool_col_r <= pool_col_r + 2;
                else begin
                    pool_col_r <= 0;
                    pool_row_r <= pool_row_r + 2;
                end
            end else begin
                $display("[V2] Layer %0d MaxPool done", layer_idx);
                seq_state <= SEQ_NEXT;
            end
        end

        // ----- NEXT: 다음 레이어 준비 -----
        SEQ_NEXT: begin
            if(layer_idx < NUM_LAYERS - 1) begin
                // pool_buf → ifm_buf 복사 (다음 레이어 입력)
                for(p = 0; p < (cur_width/2)*(cur_height/2)*cur_no; p = p+1)
                    ifm_buf[p] <= pool_buf[p];

                layer_idx  <= layer_idx + 1;
                // 다음 레이어 파라미터 설정
                cur_width  <= L_WIDTH [layer_idx+1];
                cur_height <= L_HEIGHT[layer_idx+1];
                cur_ni     <= L_NI    [layer_idx+1];
                cur_no     <= L_NO    [layer_idx+1];
                ni_cnt     <= 0;
                to_cnt     <= 0;
                relu_idx   <= 0;
                pool_row_r <= 0;
                pool_col_r <= 0;
                seq_state  <= SEQ_CONV;
                $display("[V2] Moving to Layer %0d", layer_idx+1);
            end else begin
                // 모든 레이어 완료
                seq_state <= SEQ_DONE;
                $display("[V2] All layers done! CONV00->Pool->CONV02->Pool->CONV04->Pool complete");
            end
        end

        // ----- DONE -----
        SEQ_DONE: begin
            // interrupt는 위 always 블록에서 처리
        end

        default: seq_state <= SEQ_IDLE;
        endcase
    end
end

// ============================================================
// 기존 AXI DMA (원본 유지 - IFM 초기 로드용)
// ============================================================
axi_dma_ctrl #(.BIT_TRANS(BIT_TRANS))
u_dma_ctrl(
    .clk              (clk              )
   ,.rstn             (rstn             )
   ,.i_start          (i_ctrl_reg0[0]   )
   ,.i_base_address_rd(dram_base_addr_rd)
   ,.i_base_address_wr(dram_base_addr_wr)
   ,.i_num_trans      (num_trans        )
   ,.i_max_req_blk_idx(max_req_blk_idx  )
   ,.i_read_done      (read_done        )
   ,.o_ctrl_read      (ctrl_read        )
   ,.o_read_addr      (read_addr        )
   ,.i_indata_req_wr  (indata_req_wr    )
   ,.i_write_done     (write_done       )
   ,.o_ctrl_write     (ctrl_write       )
   ,.o_write_addr     (write_addr       )
   ,.o_write_data_cnt (write_data_cnt   )
   ,.o_ctrl_write_done(ctrl_write_done  )
);

axi_dma_rd #(
    .BITS_TRANS(BIT_TRANS),
    .OUT_BITS_TRANS(OUT_BITS_TRANS),
    .AXI_WIDTH_USER(1),
    .AXI_WIDTH_ID(4),
    .AXI_WIDTH_AD(AXI_WIDTH_AD),
    .AXI_WIDTH_DA(AXI_WIDTH_DA),
    .AXI_WIDTH_DS(AXI_WIDTH_DS)
)
u_dma_read(
    .M_ARVALID  (M_ARVALID  ), .M_ARREADY  (M_ARREADY  ),
    .M_ARADDR   (M_ARADDR   ), .M_ARID     (M_ARID     ),
    .M_ARLEN    (M_ARLEN    ), .M_ARSIZE   (M_ARSIZE   ),
    .M_ARBURST  (M_ARBURST  ), .M_ARLOCK   (M_ARLOCK   ),
    .M_ARCACHE  (M_ARCACHE  ), .M_ARPROT   (M_ARPROT   ),
    .M_ARQOS    (M_ARQOS    ), .M_ARREGION (M_ARREGION ),
    .M_ARUSER   (M_ARUSER   ), .M_RVALID   (M_RVALID   ),
    .M_RREADY   (M_RREADY   ), .M_RDATA    (M_RDATA    ),
    .M_RLAST    (M_RLAST    ), .M_RID      (M_RID      ),
    .M_RUSER    (M_RUSER    ), .M_RRESP    (M_RRESP    ),
    .start_dma  (ctrl_read  ), .num_trans  (num_trans  ),
    .start_addr (read_addr  ), .data_o     (read_data  ),
    .data_vld_o (read_data_vld), .data_cnt_o(read_data_cnt),
    .done_o     (read_done  ), .clk        (clk        ),
    .rstn       (rstn       )
);

dpram_wrapper #(
    .DEPTH  (BUFF_DEPTH  ),
    .AW     (BUFF_ADDR_W ),
    .DW     (AXI_WIDTH_DA))
u_data_buffer(
    .clk    (clk          ),
    .ena    (read_data_vld),
    .addra  (read_data_cnt),
    .wea    (read_data_vld),
    .dia    (read_data    ),
    .enb    (1'd1         ),
    .addrb  (write_data_cnt),
    .dob    (write_data   )
);

axi_dma_wr #(
    .BITS_TRANS(BIT_TRANS),
    .OUT_BITS_TRANS(BIT_TRANS),
    .AXI_WIDTH_USER(1),
    .AXI_WIDTH_ID(4),
    .AXI_WIDTH_AD(AXI_WIDTH_AD),
    .AXI_WIDTH_DA(AXI_WIDTH_DA),
    .AXI_WIDTH_DS(AXI_WIDTH_DS)
)
u_dma_write(
    .M_AWID     (M_AWID     ), .M_AWADDR   (M_AWADDR   ),
    .M_AWLEN    (M_AWLEN    ), .M_AWSIZE   (M_AWSIZE   ),
    .M_AWBURST  (M_AWBURST  ), .M_AWLOCK   (M_AWLOCK   ),
    .M_AWCACHE  (M_AWCACHE  ), .M_AWPROT   (M_AWPROT   ),
    .M_AWREGION (M_AWREGION ), .M_AWQOS    (M_AWQOS    ),
    .M_AWVALID  (M_AWVALID  ), .M_AWREADY  (M_AWREADY  ),
    .M_AWUSER   (           ),
    .M_WID      (M_WID      ), .M_WDATA    (M_WDATA    ),
    .M_WSTRB    (M_WSTRB    ), .M_WLAST    (M_WLAST    ),
    .M_WVALID   (M_WVALID   ), .M_WREADY   (M_WREADY   ),
    .M_WUSER    (           ), .M_BUSER    (           ),
    .M_BID      (M_BID      ), .M_BRESP    (M_BRESP    ),
    .M_BVALID   (M_BVALID   ), .M_BREADY   (M_BREADY   ),
    .start_dma  (ctrl_write ), .num_trans  (num_trans  ),
    .start_addr (write_addr ), .indata     (write_data ),
    .indata_req_o(indata_req_wr), .done_o  (write_done ),
    .fail_check (           ), .clk        (clk        ),
    .rstn       (rstn       )
);

endmodule
