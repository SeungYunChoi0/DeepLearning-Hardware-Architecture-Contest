`timescale 1ns / 1ns

module yolo_engine_tb;

localparam MEM_ADDRW = 22;
localparam MEM_DW    = 16;
localparam A = 32;
localparam D = 32;
localparam I = 4;
localparam L = 8;
localparam M = D/8;

localparam IFM_WIDTH0   = 256;
localparam IFM_HEIGHT0  = 256;
localparam IFM_PIXELS0  = IFM_WIDTH0*IFM_HEIGHT0;
localparam POOL_WIDTH0  = 128;
localparam POOL_HEIGHT0 = 128;
localparam WATCHDOG_CYCLES = 20_000_000;

localparam IFM_FILE_16B      = "CONV00_input_16b.hex";
localparam CONV_OUTPUT_IMG00 = "./CONV00_output_ch00.bmp";
localparam CONV_OUTPUT_IMG01 = "./CONV00_output_ch01.bmp";
localparam CONV_OUTPUT_IMG02 = "./CONV00_output_ch02.bmp";
localparam CONV_OUTPUT_IMG03 = "./CONV00_output_ch03.bmp";

parameter CLK_PERIOD = 10;
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
wire                 mem_we;
wire [MEM_DW-1:0]    mem_di;
wire [MEM_DW-1:0]    mem_do;

wire        read_data_vld;
wire [31:0] read_data;
wire        network_done;
wire        network_done_led;

// ============================================================
// AXI SRAM 모델 (DMA용 메모리 모델)
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

// yolo_engine 내부는 별도 $readmemh를 쓰지만,
// DMA 경로에서 X 전파를 막기 위해 외부 메모리 모델도 유지한다.
sram #(
    .FILE_NAME(IFM_FILE_16B),
    .SIZE     (2**MEM_ADDRW),
    .WL_ADDR  (MEM_ADDRW),
    .WL_DATA  (MEM_DW))
u_ext_mem_input(
    .clk  (clk     ),
    .rst  (rstn    ),
    .addr (mem_addr),
    .wdata(mem_di  ),
    .rdata(mem_do  ),
    .ena  (mem_we  )
);

// ============================================================
// DUT
// ============================================================
reg [31:0] i_0;
reg [31:0] i_1;
reg [31:0] i_2;
reg [31:0] i_3;

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
// network_done 이후 내부 ofm_buf를 직접 스캔해서 BMP 저장
// ============================================================
reg        dump_active;
reg [16:0] dump_idx;
reg [7:0]  dump_ch0, dump_ch1, dump_ch2, dump_ch3;
reg        dump_vld;
wire       dump_done0, dump_done1, dump_done2, dump_done3;

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG00), .WIDTH(IFM_WIDTH0), .HEIGHT(IFM_HEIGHT0))
u_bmp_conv00_ch0(
    .clk(clk), .rstn(rstn), .din(dump_ch0), .vld(dump_vld), .frame_done(dump_done0));

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG01), .WIDTH(IFM_WIDTH0), .HEIGHT(IFM_HEIGHT0))
u_bmp_conv00_ch1(
    .clk(clk), .rstn(rstn), .din(dump_ch1), .vld(dump_vld), .frame_done(dump_done1));

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG02), .WIDTH(IFM_WIDTH0), .HEIGHT(IFM_HEIGHT0))
u_bmp_conv00_ch2(
    .clk(clk), .rstn(rstn), .din(dump_ch2), .vld(dump_vld), .frame_done(dump_done2));

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG03), .WIDTH(IFM_WIDTH0), .HEIGHT(IFM_HEIGHT0))
u_bmp_conv00_ch3(
    .clk(clk), .rstn(rstn), .din(dump_ch3), .vld(dump_vld), .frame_done(dump_done3));

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dump_active <= 1'b0;
        dump_idx    <= 17'd0;
        dump_vld    <= 1'b0;
        dump_ch0    <= 8'd0;
        dump_ch1    <= 8'd0;
        dump_ch2    <= 8'd0;
        dump_ch3    <= 8'd0;
    end else begin
        dump_vld <= 1'b0;

        if (network_done && !dump_active && dump_idx == 0) begin
            dump_active <= 1'b1;
            $display("[TB] network_done 감지, CONV00 ofm_buf ch0~3 BMP dump 시작");
        end

        if (dump_active) begin
            dump_ch0 <= u_yolo_engine.ofm_buf[(0*IFM_PIXELS0) + dump_idx];
            dump_ch1 <= u_yolo_engine.ofm_buf[(1*IFM_PIXELS0) + dump_idx];
            dump_ch2 <= u_yolo_engine.ofm_buf[(2*IFM_PIXELS0) + dump_idx];
            dump_ch3 <= u_yolo_engine.ofm_buf[(3*IFM_PIXELS0) + dump_idx];
            dump_vld <= 1'b1;

            if (dump_idx == IFM_PIXELS0-1) begin
                dump_idx    <= dump_idx;
                dump_active <= 1'b0;
            end else begin
                dump_idx <= dump_idx + 1'b1;
            end
        end
    end
end

// ============================================================
// 시뮬레이션 제어 + watchdog
// ============================================================
integer watchdog_count;
initial begin
    rstn = 1'b0;
    i_0  = 32'd0;
    i_1  = 32'd0;
    i_2  = (4*256*256)*4;
    i_3  = 32'd0;
    watchdog_count = 0;

    #(4*CLK_PERIOD);
    rstn = 1'b1;

    #(100*CLK_PERIOD);
    @(posedge clk);

    i_0 = 32'd1;
    $display("[TB] network_start 신호 전송");

    #(10*CLK_PERIOD);
    @(posedge clk);
    i_0 = 32'd0;

    $display("[TB] network_done 대기 중...");
end

always @(posedge clk) begin
    if (rstn) begin
        watchdog_count <= watchdog_count + 1;

        if (network_done) begin
            $display("[TB] network_done! time=%0t", $time);
        end

        if (dump_done0 && dump_done1 && dump_done2 && dump_done3) begin
            $display("[TB] BMP dump 완료. 시뮬레이션 종료");
            #(20*CLK_PERIOD);
            $finish;
        end

        if (watchdog_count > WATCHDOG_CYCLES) begin
            $display("[TB][ERROR] watchdog timeout. network_done 미도달");
            $finish;
        end
    end
end

endmodule
