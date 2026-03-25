//----------------------------------------------------------------+
// Project: Deep Learning Hardware Accelerator Design Contest
// Module:  yolo_engine_tb (V3)
// Description:
//   CONV00 -> MaxPool -> CONV02 -> MaxPool -> CONV04 -> MaxPool
//   결과를 bmp로 저장하여 확인
//
// 시뮬레이션 실행 전 준비사항:
//   Vivado sim_1에 Add Sources (Simulation Sources):
//     - CONV00_input_32b.hex
//     - CONV00_param_weight.hex
//     - CONV02_param_weight.hex
//     - CONV04_param_weight.hex
//   inout_data_hw 폴더 생성:
//     yolohw/sim/inout_data_hw/
//----------------------------------------------------------------+
`timescale 1ns / 1ns

module yolo_engine_tb;

`include "user_param_h.v"

localparam MEM_ADDRW = 22;
localparam MEM_DW    = 16;
localparam A = 32;
localparam D = 32;
localparam I = 4;
localparam L = 8;
localparam M = D/8;

// Clock
parameter CLK_PERIOD = 10;  // 100MHz
reg clk;
reg rstn;

initial begin
    clk = 1'b1;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// ============================================================
// AXI 인터페이스 신호
// ============================================================
wire [I-1:0] M_AWID;
wire [A-1:0] M_AWADDR;
wire [L-1:0] M_AWLEN;
wire [2:0]   M_AWSIZE;
wire [1:0]   M_AWBURST;
wire [1:0]   M_AWLOCK;
wire [3:0]   M_AWCACHE;
wire [2:0]   M_AWPROT;
wire         M_AWVALID;
wire         M_AWREADY;
wire [I-1:0] M_WID;
wire [D-1:0] M_WDATA;
wire [M-1:0] M_WSTRB;
wire         M_WLAST;
wire         M_WVALID;
wire         M_WREADY;
wire [I-1:0] M_BID;
wire [1:0]   M_BRESP;
wire         M_BVALID;
wire         M_BREADY;
wire [I-1:0] M_ARID;
wire [A-1:0] M_ARADDR;
wire [L-1:0] M_ARLEN;
wire [2:0]   M_ARSIZE;
wire [1:0]   M_ARBURST;
wire [1:0]   M_ARLOCK;
wire [3:0]   M_ARCACHE;
wire [2:0]   M_ARPROT;
wire         M_ARVALID;
wire         M_ARREADY;
wire [I-1:0] M_RID;
wire [D-1:0] M_RDATA;
wire [1:0]   M_RRESP;
wire         M_RLAST;
wire         M_RVALID;
wire         M_RREADY;

wire [MEM_ADDRW-1:0] mem_addr;
wire                  mem_we;
wire [MEM_DW-1:0]    mem_di;
wire [MEM_DW-1:0]    mem_do;

wire        i_WVALID    = M_WVALID;
wire [31:0] i_WDATA     = M_WDATA;
wire        read_data_vld;
wire [31:0] read_data;
wire        network_done;
wire        network_done_led;

// ============================================================
// AXI SRAM 모델 (DRAM 역할)
// yolo_engine의 DMA Read 요청에 응답
// ============================================================
axi_sram_if #(
    .MEM_ADDRW(MEM_ADDRW), .MEM_DW(MEM_DW),
    .A(A), .I(I), .L(L), .D(D), .M(M))
u_axi_ext_mem_if_input(
    .ACLK(clk), .ARESETn(rstn),
    .AWID(M_AWID), .AWADDR(M_AWADDR), .AWLEN(M_AWLEN),
    .AWSIZE(M_AWSIZE), .AWBURST(M_AWBURST), .AWLOCK(M_AWLOCK),
    .AWCACHE(M_AWCACHE), .AWPROT(M_AWPROT), .AWVALID(M_AWVALID),
    .AWREADY(M_AWREADY),
    .WID(M_WID), .WDATA(M_WDATA), .WSTRB(M_WSTRB),
    .WLAST(M_WLAST), .WVALID(M_WVALID), .WREADY(M_WREADY),
    .BID(M_BID), .BRESP(M_BRESP), .BVALID(M_BVALID), .BREADY(M_BREADY),
    .ARID(M_ARID), .ARADDR(M_ARADDR), .ARLEN(M_ARLEN),
    .ARSIZE(M_ARSIZE), .ARBURST(M_ARBURST), .ARLOCK(M_ARLOCK),
    .ARCACHE(M_ARCACHE), .ARPROT(M_ARPROT), .ARVALID(M_ARVALID),
    .ARREADY(M_ARREADY),
    .RID(M_RID), .RDATA(M_RDATA), .RRESP(M_RRESP),
    .RLAST(M_RLAST), .RVALID(M_RVALID), .RREADY(M_RREADY),
    .mem_addr(mem_addr), .mem_we(mem_we),
    .mem_di(mem_di), .mem_do(mem_do)
);

// SRAM 모델: IFM_FILE 로드 (DRAM 초기화용)
// yolo_engine.v에서 $readmemh로 직접 로드하므로 여기선 더미
sram #(
    .FILE_NAME(IFM_FILE),
    .SIZE     (2**MEM_ADDRW),
    .WL_ADDR  (MEM_ADDRW),
    .WL_DATA  (MEM_DW))
u_ext_mem_input(
    .clk  (clk     ),
    .rst  (rstn    ),
    .addr (mem_addr),
    .wdata(mem_di  ),
    .rdata(mem_do  ),
    .ena  (1'b0    )  // Read only
);

// ============================================================
// DUT: yolo_engine
// ============================================================
reg [31:0] i_0 = 0;
reg [31:0] i_1 = 0;
reg [31:0] i_2 = 0;
reg [31:0] i_3 = 0;

yolo_engine #(
    .AXI_WIDTH_AD(A), .AXI_WIDTH_ID(4),
    .AXI_WIDTH_DA(D), .AXI_WIDTH_DS(M),
    .MEM_BASE_ADDR(2048), .MEM_DATA_BASE_ADDR(2048))
u_yolo_engine(
    .clk(clk), .rstn(rstn),
    .i_ctrl_reg0(i_0), .i_ctrl_reg1(i_1),
    .i_ctrl_reg2(i_2), .i_ctrl_reg3(i_3),
    .M_ARVALID(M_ARVALID), .M_ARREADY(M_ARREADY),
    .M_ARADDR (M_ARADDR ), .M_ARID   (M_ARID   ),
    .M_ARLEN  (M_ARLEN  ), .M_ARSIZE (M_ARSIZE ),
    .M_ARBURST(M_ARBURST), .M_ARLOCK (M_ARLOCK ),
    .M_ARCACHE(M_ARCACHE), .M_ARPROT (M_ARPROT ),
    .M_ARQOS  (         ), .M_ARREGION(        ),
    .M_ARUSER (         ), .M_RVALID (M_RVALID ),
    .M_RREADY (M_RREADY ), .M_RDATA  (M_RDATA  ),
    .M_RLAST  (M_RLAST  ), .M_RID    (M_RID    ),
    .M_RUSER  (         ), .M_RRESP  (M_RRESP  ),
    .M_AWVALID(M_AWVALID), .M_AWREADY(M_AWREADY),
    .M_AWADDR (M_AWADDR ), .M_AWID   (M_AWID   ),
    .M_AWLEN  (M_AWLEN  ), .M_AWSIZE (M_AWSIZE ),
    .M_AWBURST(M_AWBURST), .M_AWLOCK (M_AWLOCK ),
    .M_AWCACHE(M_AWCACHE), .M_AWPROT (M_AWPROT ),
    .M_AWQOS  (         ), .M_AWREGION(        ),
    .M_AWUSER (         ),
    .M_WVALID (M_WVALID ), .M_WREADY (M_WREADY ),
    .M_WDATA  (M_WDATA  ), .M_WSTRB  (M_WSTRB  ),
    .M_WLAST  (M_WLAST  ), .M_WID    (M_WID    ),
    .M_WUSER  (         ),
    .M_BVALID (M_BVALID ), .M_BREADY (M_BREADY ),
    .M_BRESP  (M_BRESP  ), .M_BID    (M_BID    ),
    .M_BUSER  (         ),
    .network_done    (network_done    ),
    .network_done_led(network_done_led),
    .read_data_vld   (read_data_vld   ),
    .read_data       (read_data       )
);

// ============================================================
// 시뮬레이션 제어
// ============================================================
initial begin
    rstn = 1'b0;
    i_0  = 0; i_1 = 0;
    i_2  = (4*256*256)*4;  // Write base address
    i_3  = 0;

    // 리셋 해제
    #(4*CLK_PERIOD) rstn = 1'b1;

    // 초기화 대기 ($readmemh 완료 대기)
    #(100*CLK_PERIOD)
    @(posedge clk);

    // network_start (i_0[0]=1)
    i_0 = 32'd1;
    $display("[TB] network_start 신호 전송");

    #(10*CLK_PERIOD)
    @(posedge clk);
    i_0 = 32'd0;

    // network_done 대기
    $display("[TB] network_done 대기 중...");
    while(!network_done) begin
        #(1000*CLK_PERIOD) @(posedge clk);
    end

    $display("[TB] network_done! 시뮬레이션 완료");
    #(100*CLK_PERIOD)
    @(posedge clk) $stop;
end

// ============================================================
// bmp 저장 - CONV00 출력 (ofm_buf에서 직접 읽음)
// vld_o 신호를 yolo_engine 내부에서 끌어올 수 없으므로
// network_done 후 ofm_buf를 덤프하는 방식 대신
// DMA Write 데이터 스트림을 활용
// ============================================================

// CONV00 출력 bmp (DMA Write 스트림 활용 - 4바이트씩)
`ifdef CHECK_DMA_WRITE
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG00),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_conv00_ch0(
    .clk(clk), .rstn(rstn),
    .din(i_WDATA[7:0]), .vld(i_WVALID), .frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG01),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_conv00_ch1(
    .clk(clk), .rstn(rstn),
    .din(i_WDATA[15:8]), .vld(i_WVALID), .frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG02),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_conv00_ch2(
    .clk(clk), .rstn(rstn),
    .din(i_WDATA[23:16]), .vld(i_WVALID), .frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG03),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_conv00_ch3(
    .clk(clk), .rstn(rstn),
    .din(i_WDATA[31:24]), .vld(i_WVALID), .frame_done());
`else
// DMA Read 확인용 (입력 이미지 덤프)
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG00),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_input_ch0(
    .clk(clk), .rstn(rstn),
    .din(read_data[7:0]), .vld(read_data_vld), .frame_done());
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG01),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_input_ch1(
    .clk(clk), .rstn(rstn),
    .din(read_data[15:8]), .vld(read_data_vld), .frame_done());
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG02),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_input_ch2(
    .clk(clk), .rstn(rstn),
    .din(read_data[23:16]), .vld(read_data_vld), .frame_done());
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG03),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
u_bmp_input_ch3(
    .clk(clk), .rstn(rstn),
    .din(read_data[31:24]), .vld(read_data_vld), .frame_done());
`endif

// ============================================================
// ofm_buf 직접 접근 bmp 저장
// yolo_engine의 ofm_buf를 계층적 참조로 접근
// ============================================================
// CONV00 출력 bmp (ofm_buf 직접 덤프 - network_done 후)
reg        dump_en;
reg [31:0] dump_idx;
reg [1:0]  dump_layer;  // 0=CONV00, 1=CONV02, 2=CONV04
reg [11:0] dump_w, dump_h;
reg [6:0]  dump_no;

// CONV00 출력 채널 0~3 bmp
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG00),.WIDTH(256),.HEIGHT(256))
u_ofm_conv00_ch0(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.ofm_buf[dump_idx]),
    .vld(dump_en && dump_layer==0), .frame_done());

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG01),.WIDTH(256),.HEIGHT(256))
u_ofm_conv00_ch1(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.ofm_buf[dump_idx + 256*256]),
    .vld(dump_en && dump_layer==0), .frame_done());

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG02),.WIDTH(256),.HEIGHT(256))
u_ofm_conv00_ch2(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.ofm_buf[dump_idx + 2*256*256]),
    .vld(dump_en && dump_layer==0), .frame_done());

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG03),.WIDTH(256),.HEIGHT(256))
u_ofm_conv00_ch3(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.ofm_buf[dump_idx + 3*256*256]),
    .vld(dump_en && dump_layer==0), .frame_done());

// ofm_buf 덤프 제어
// network_done 후 순서대로 CONV00 → CONV02 → CONV04 출력 저장
always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        dump_en    <= 0;
        dump_idx   <= 0;
        dump_layer <= 0;
        dump_w     <= 256;
        dump_h     <= 256;
        dump_no    <= 16;
    end else begin
        if(network_done && !dump_en) begin
            dump_en    <= 1;
            dump_idx   <= 0;
            dump_layer <= 0;
            dump_w     <= 256;
            dump_h     <= 256;
            dump_no    <= 16;
            $display("[TB] ofm_buf 덤프 시작 (CONV00 출력)");
        end else if(dump_en) begin
            if(dump_idx < dump_w * dump_h - 1) begin
                dump_idx <= dump_idx + 1;
            end else begin
                dump_en <= 0;
                $display("[TB] bmp 저장 완료");
            end
        end
    end
end

endmodule
