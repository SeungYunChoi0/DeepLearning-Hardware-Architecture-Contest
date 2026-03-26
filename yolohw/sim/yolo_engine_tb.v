//----------------------------------------------------------------+
// Project: AIX2026 Deep Learning Hardware Accelerator Design Contest
// Module:  yolo_engine_tb (V5)
// Description:
//   CONV00 -> MaxPool -> CONV02 -> MaxPool -> CONV04 -> MaxPool
//
// BMP 출력:
//   각 레이어 SEQ_RELU->SEQ_MAXPOOL 전이 시점에 ofm_buf 직접 스캔
//   CONV00: 256x256 ch0~3  (CONV_OUTPUT_IMG00~03)
//   CONV02: 128x128 ch0~1  (CONV_OUTPUT_IMG04~05)
//   CONV04:  64x64  ch0~1  (CONV_OUTPUT_IMG06~07)
//
// SW vs HW 검증:
//   yolo_engine.v $writememh → CONV0X_hw_ofm.hex
//   C코드 출력                → CONV0X_output.hex
//
// sim_1 Add Sources (hex 파일):
//   [필수] CONV00_input.hex
//   [필수] CONV00/02/04_param_weight.hex
//   [필수] CONV00/02/04_param_biases.hex
//   [검증] CONV00/02/04_output.hex
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
wire                  mem_we;
wire [MEM_DW-1:0]    mem_di;
wire [MEM_DW-1:0]    mem_do;

wire        read_data_vld;
wire [31:0] read_data;
wire        network_done;
wire        network_done_led;

// ============================================================
// AXI SRAM 모델 (DMA X 전파 방지용 더미)
// ============================================================
axi_sram_if #(
    .MEM_ADDRW(MEM_ADDRW), .MEM_DW(MEM_DW),
    .A(A), .I(I), .L(L), .D(D), .M(M))
u_axi_ext_mem_if_input(
    .ACLK(clk), .ARESETn(rstn),
    .AWID(M_AWID),   .AWADDR(M_AWADDR), .AWLEN(M_AWLEN),
    .AWSIZE(M_AWSIZE),.AWBURST(M_AWBURST),.AWLOCK(M_AWLOCK),
    .AWCACHE(M_AWCACHE),.AWPROT(M_AWPROT),.AWVALID(M_AWVALID),
    .AWREADY(M_AWREADY),
    .WID(M_WID),  .WDATA(M_WDATA), .WSTRB(M_WSTRB),
    .WLAST(M_WLAST),.WVALID(M_WVALID),.WREADY(M_WREADY),
    .BID(M_BID),  .BRESP(M_BRESP), .BVALID(M_BVALID),.BREADY(M_BREADY),
    .ARID(M_ARID), .ARADDR(M_ARADDR),.ARLEN(M_ARLEN),
    .ARSIZE(M_ARSIZE),.ARBURST(M_ARBURST),.ARLOCK(M_ARLOCK),
    .ARCACHE(M_ARCACHE),.ARPROT(M_ARPROT),.ARVALID(M_ARVALID),
    .ARREADY(M_ARREADY),
    .RID(M_RID),  .RDATA(M_RDATA), .RRESP(M_RRESP),
    .RLAST(M_RLAST),.RVALID(M_RVALID),.RREADY(M_RREADY),
    .mem_addr(mem_addr),.mem_we(mem_we),
    .mem_di(mem_di),.mem_do(mem_do)
);

sram #(
    .FILE_NAME(IFM_FILE),
    .SIZE     (2**MEM_ADDRW),
    .WL_ADDR  (MEM_ADDRW),
    .WL_DATA  (MEM_DW))
u_ext_mem_input(
    .clk  (clk),  .rst  (rstn),
    .addr (mem_addr), .wdata(mem_di),
    .rdata(mem_do),   .ena  (1'b0)
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
// 레이어 완료 감지
// SEQ_RELU(4d4) → SEQ_MAXPOOL(4d5) 전이 = $writememh 직후 1클럭
// ============================================================
wire [3:0] seq_state = u_yolo_engine.seq_state;
wire [1:0] layer_idx = u_yolo_engine.layer_idx;

reg [3:0] seq_state_d;
always @(posedge clk) seq_state_d <= seq_state;

wire layer_relu_done = (seq_state_d == 4'd4) && (seq_state == 4'd5);
wire conv00_done     = layer_relu_done && (layer_idx == 2'd0);
wire conv02_done     = layer_relu_done && (layer_idx == 2'd1);
wire conv04_done     = layer_relu_done && (layer_idx == 2'd2);

// ============================================================
// BMP dump - CONV00 (256x256, ch0~3)
// ============================================================
reg        dump00_active;
reg [16:0] dump00_idx;
reg        dump00_vld;
reg [7:0]  d00_ch0, d00_ch1, d00_ch2, d00_ch3;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dump00_active<=0; dump00_idx<=0; dump00_vld<=0;
    end else begin
        dump00_vld <= 0;
        if (conv00_done && !dump00_active) begin
            dump00_active <= 1; dump00_idx <= 0;
            $display("[TB] CONV00 BMP dump 시작 (256x256 ch0~3)");
        end
        if (dump00_active) begin
            d00_ch0 <= u_yolo_engine.ofm_buf[0*256*256 + dump00_idx];
            d00_ch1 <= u_yolo_engine.ofm_buf[1*256*256 + dump00_idx];
            d00_ch2 <= u_yolo_engine.ofm_buf[2*256*256 + dump00_idx];
            d00_ch3 <= u_yolo_engine.ofm_buf[3*256*256 + dump00_idx];
            dump00_vld <= 1;
            if (dump00_idx == 256*256-1) begin
                dump00_active <= 0;
                $display("[TB] CONV00 BMP dump 완료");
            end else
                dump00_idx <= dump00_idx + 1;
        end
    end
end

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG00),.WIDTH(256),.HEIGHT(256))
u_bmp00_ch0(.clk(clk),.rstn(rstn),.din(d00_ch0),.vld(dump00_vld),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG01),.WIDTH(256),.HEIGHT(256))
u_bmp00_ch1(.clk(clk),.rstn(rstn),.din(d00_ch1),.vld(dump00_vld),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG02),.WIDTH(256),.HEIGHT(256))
u_bmp00_ch2(.clk(clk),.rstn(rstn),.din(d00_ch2),.vld(dump00_vld),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG03),.WIDTH(256),.HEIGHT(256))
u_bmp00_ch3(.clk(clk),.rstn(rstn),.din(d00_ch3),.vld(dump00_vld),.frame_done());

// ============================================================
// BMP dump - CONV02 (128x128, ch0~1)
// ============================================================
reg        dump02_active;
reg [13:0] dump02_idx;
reg        dump02_vld;
reg [7:0]  d02_ch0, d02_ch1;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dump02_active<=0; dump02_idx<=0; dump02_vld<=0;
    end else begin
        dump02_vld <= 0;
        if (conv02_done && !dump02_active) begin
            dump02_active <= 1; dump02_idx <= 0;
            $display("[TB] CONV02 BMP dump 시작 (128x128 ch0~1)");
        end
        if (dump02_active) begin
            d02_ch0 <= u_yolo_engine.ofm_buf[0*128*128 + dump02_idx];
            d02_ch1 <= u_yolo_engine.ofm_buf[1*128*128 + dump02_idx];
            dump02_vld <= 1;
            if (dump02_idx == 128*128-1) begin
                dump02_active <= 0;
                $display("[TB] CONV02 BMP dump 완료");
            end else
                dump02_idx <= dump02_idx + 1;
        end
    end
end

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG04),.WIDTH(128),.HEIGHT(128))
u_bmp02_ch0(.clk(clk),.rstn(rstn),.din(d02_ch0),.vld(dump02_vld),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG05),.WIDTH(128),.HEIGHT(128))
u_bmp02_ch1(.clk(clk),.rstn(rstn),.din(d02_ch1),.vld(dump02_vld),.frame_done());

// ============================================================
// BMP dump - CONV04 (64x64, ch0~1)
// ============================================================
reg        dump04_active;
reg [11:0] dump04_idx;
reg        dump04_vld;
reg [7:0]  d04_ch0, d04_ch1;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        dump04_active<=0; dump04_idx<=0; dump04_vld<=0;
    end else begin
        dump04_vld <= 0;
        if (conv04_done && !dump04_active) begin
            dump04_active <= 1; dump04_idx <= 0;
            $display("[TB] CONV04 BMP dump 시작 (64x64 ch0~1)");
        end
        if (dump04_active) begin
            d04_ch0 <= u_yolo_engine.ofm_buf[0*64*64 + dump04_idx];
            d04_ch1 <= u_yolo_engine.ofm_buf[1*64*64 + dump04_idx];
            dump04_vld <= 1;
            if (dump04_idx == 64*64-1) begin
                dump04_active <= 0;
                $display("[TB] CONV04 BMP dump 완료");
            end else
                dump04_idx <= dump04_idx + 1;
        end
    end
end

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG06),.WIDTH(64),.HEIGHT(64))
u_bmp04_ch0(.clk(clk),.rstn(rstn),.din(d04_ch0),.vld(dump04_vld),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG07),.WIDTH(64),.HEIGHT(64))
u_bmp04_ch1(.clk(clk),.rstn(rstn),.din(d04_ch1),.vld(dump04_vld),.frame_done());

// ============================================================
// SW vs HW 검증 (network_done 이후)
// ============================================================
integer vfy_fd_sw, vfy_fd_hw;
integer vfy_err, vfy_total, vfy_ii;
reg [7:0] sw_val, hw_val;

initial begin : VERIFY
    wait(network_done);
    #(10*CLK_PERIOD);

    // CONV00: 16ch x 256x256
    vfy_err=0; vfy_total=16*256*256;
    vfy_fd_sw=$fopen("CONV00_output.hex","r");
    vfy_fd_hw=$fopen("CONV00_hw_ofm.hex","r");
    if(vfy_fd_sw==0||vfy_fd_hw==0)
        $display("[VERIFY] CONV00: 파일 없음 - 스킵");
    else begin
        for(vfy_ii=0;vfy_ii<vfy_total;vfy_ii=vfy_ii+1) begin
            $fscanf(vfy_fd_sw,"%h\n",sw_val);
            $fscanf(vfy_fd_hw,"%h\n",hw_val);
            if(sw_val!==hw_val) vfy_err=vfy_err+1;
        end
        $fclose(vfy_fd_sw); $fclose(vfy_fd_hw);
        $display("[VERIFY] CONV00 OFM: total=%0d, errors=%0d -> %s",
                 vfy_total, vfy_err, (vfy_err==0)?"PASS":"FAIL");
    end

    // CONV02: 32ch x 128x128
    vfy_err=0; vfy_total=32*128*128;
    vfy_fd_sw=$fopen("CONV02_output.hex","r");
    vfy_fd_hw=$fopen("CONV02_hw_ofm.hex","r");
    if(vfy_fd_sw==0||vfy_fd_hw==0)
        $display("[VERIFY] CONV02: 파일 없음 - 스킵");
    else begin
        for(vfy_ii=0;vfy_ii<vfy_total;vfy_ii=vfy_ii+1) begin
            $fscanf(vfy_fd_sw,"%h\n",sw_val);
            $fscanf(vfy_fd_hw,"%h\n",hw_val);
            if(sw_val!==hw_val) vfy_err=vfy_err+1;
        end
        $fclose(vfy_fd_sw); $fclose(vfy_fd_hw);
        $display("[VERIFY] CONV02 OFM: total=%0d, errors=%0d -> %s",
                 vfy_total, vfy_err, (vfy_err==0)?"PASS":"FAIL");
    end

    // CONV04: 64ch x 64x64
    vfy_err=0; vfy_total=64*64*64;
    vfy_fd_sw=$fopen("CONV04_output.hex","r");
    vfy_fd_hw=$fopen("CONV04_hw_ofm.hex","r");
    if(vfy_fd_sw==0||vfy_fd_hw==0)
        $display("[VERIFY] CONV04: 파일 없음 - 스킵");
    else begin
        for(vfy_ii=0;vfy_ii<vfy_total;vfy_ii=vfy_ii+1) begin
            $fscanf(vfy_fd_sw,"%h\n",sw_val);
            $fscanf(vfy_fd_hw,"%h\n",hw_val);
            if(sw_val!==hw_val) vfy_err=vfy_err+1;
        end
        $fclose(vfy_fd_sw); $fclose(vfy_fd_hw);
        $display("[VERIFY] CONV04 OFM: total=%0d, errors=%0d -> %s",
                 vfy_total, vfy_err, (vfy_err==0)?"PASS":"FAIL");
    end

    $display("[VERIFY] 전체 검증 완료");
end

// ============================================================
// 시뮬레이션 제어
// ============================================================
initial begin
    rstn=1'b0; i_0=0; i_1=0; i_2=(4*256*256)*4; i_3=0;

    #(4*CLK_PERIOD);  rstn=1'b1;
    #(100*CLK_PERIOD); @(posedge clk);

    i_0=32'd1;
    $display("[TB] network_start 신호 전송");
    #(10*CLK_PERIOD); @(posedge clk);
    i_0=32'd0;

    $display("[TB] network_done 대기 중...");
    wait(network_done);
    $display("[TB] network_done! time=%0t", $time);

    // BMP dump 완료 대기
    wait(!dump00_active && !dump02_active && !dump04_active);
    #(200*CLK_PERIOD);
    $display("[TB] 시뮬레이션 종료");
    $stop;
end

endmodule
