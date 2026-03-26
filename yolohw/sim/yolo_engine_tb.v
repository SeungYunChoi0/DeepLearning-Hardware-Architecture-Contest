//----------------------------------------------------------------+
// Project: Deep Learning Hardware Accelerator Design Contest
// Module:  yolo_engine_tb (V4 - 수정본)
// Description:
//   CONV00 -> MaxPool -> CONV02 -> MaxPool -> CONV04 -> MaxPool
//   결과를 bmp로 저장 + SW vs HW 수치 비교 검증
//
// [수정사항]
//   1. dump 제어: CONV00 전용 -> 3레이어 순차 dump
//   2. bmp_image_writer: CONV02(128x128), CONV04(64x64) 추가
//   3. SW vs HW 검증 블록 추가 (CONV00/02/04 output hex 비교)
//
// 시뮬레이션 실행 전 준비사항:
//   Vivado sim_1 -> Add Sources (Simulation Sources):
//     [필수] CONV00_input.hex
//     [필수] CONV00/02/04_param_weight.hex
//     [필수] CONV00/02/04_param_biases.hex
//     [검증] CONV00/02/04_output.hex  (SW 기준값, C코드 생성)
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

// SRAM 모델 (yolo_engine.v에서 $readmemh로 직접 로드하므로 더미)
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

    // 초기화 대기
    #(10*CLK_PERIOD)
    @(posedge clk);

    // network_start
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

    $display("[TB] network_done! bmp dump 시작...");
    // dump 완료 대기 (dump_en이 0이 될 때까지)
    wait(dump_en == 0);
    $display("[TB] 전체 시뮬레이션 완료");
    // SW vs HW 검증 완료 대기
    #(200*CLK_PERIOD)
    @(posedge clk) $stop;
end

// ============================================================
// [수정] ofm_buf / pool_buf 순차 dump 제어
//   dump_layer 0: CONV00 ofm_buf  (256x256, ch0~3)
//   dump_layer 1: CONV02 pool_buf (128x128, ch0~1)
//   dump_layer 2: CONV04 pool_buf (64x64,   ch0~1)
// ============================================================
reg        dump_en;
reg [31:0] dump_idx;
reg [1:0]  dump_layer;

always @(posedge clk, negedge rstn) begin
    if(!rstn) begin
        dump_en    <= 0;
        dump_idx   <= 0;
        dump_layer <= 0;
    end else begin
        if(network_done && !dump_en) begin
            dump_en    <= 1;
            dump_idx   <= 0;
            dump_layer <= 0;
            $display("[TB] CONV00 ofm dump 시작 (256x256)");
        end else if(dump_en) begin
            case(dump_layer)
            // CONV00 출력: 256x256 픽셀
            0: begin
                if(dump_idx < 256*256-1)
                    dump_idx <= dump_idx + 1;
                else begin
                    dump_idx   <= 0;
                    dump_layer <= 1;
                    $display("[TB] CONV02 pool dump 시작 (128x128)");
                end
            end
            // CONV02 MaxPool 후: 128x128 픽셀
            1: begin
                if(dump_idx < 128*128-1)
                    dump_idx <= dump_idx + 1;
                else begin
                    dump_idx   <= 0;
                    dump_layer <= 2;
                    $display("[TB] CONV04 pool dump 시작 (64x64)");
                end
            end
            // CONV04 MaxPool 후: 64x64 픽셀
            2: begin
                if(dump_idx < 64*64-1)
                    dump_idx <= dump_idx + 1;
                else begin
                    dump_en <= 0;
                    $display("[TB] 전체 bmp dump 완료");
                end
            end
            default: dump_en <= 0;
            endcase
        end
    end
end

// ============================================================
// bmp 저장 - CONV00 출력 (256x256, ofm_buf ch0~3)
// ============================================================
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG00),.WIDTH(256),.HEIGHT(256))
u_ofm_conv00_ch0(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.ofm_buf[dump_idx + 0*256*256]),
    .vld(dump_en && dump_layer==0), .frame_done());

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG01),.WIDTH(256),.HEIGHT(256))
u_ofm_conv00_ch1(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.ofm_buf[dump_idx + 1*256*256]),
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

// ============================================================
// [수정] bmp 저장 - CONV02 MaxPool 후 (128x128, pool_buf ch0~1)
// ============================================================
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG04),.WIDTH(128),.HEIGHT(128))
u_ofm_conv02_ch0(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.pool_buf[dump_idx + 0*128*128]),
    .vld(dump_en && dump_layer==1), .frame_done());

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG05),.WIDTH(128),.HEIGHT(128))
u_ofm_conv02_ch1(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.pool_buf[dump_idx + 1*128*128]),
    .vld(dump_en && dump_layer==1), .frame_done());

// ============================================================
// [수정] bmp 저장 - CONV04 MaxPool 후 (64x64, pool_buf ch0~1)
// ============================================================
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG06),.WIDTH(64),.HEIGHT(64))
u_ofm_conv04_ch0(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.pool_buf[dump_idx + 0*64*64]),
    .vld(dump_en && dump_layer==2), .frame_done());

bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG07),.WIDTH(64),.HEIGHT(64))
u_ofm_conv04_ch1(
    .clk(clk), .rstn(rstn),
    .din(u_yolo_engine.pool_buf[dump_idx + 1*64*64]),
    .vld(dump_en && dump_layer==2), .frame_done());

// ============================================================
// SW vs HW 비교 검증
//
// 비교 대상:
//   SW: CONV0X_output.hex  (C코드 생성, MaxPool 전 OFM, requant된 INT8)
//   HW: CONV0X_hw_ofm.hex  (yolo_engine.v에서 $writememh로 저장, MaxPool 전 OFM)
//
// 두 파일 모두 MaxPool 전 OFM이므로 직접 비교 가능
// ±1 차이는 정수 시프트 vs 부동소수점 나눗셈 반올림 차이로 허용
//
// sim_1에 추가 필요:
//   CONV00_output.hex / CONV02_output.hex / CONV04_output.hex
// ============================================================
integer vfy_fd_sw, vfy_fd_hw;
integer vfy_err;
integer vfy_total;
integer vfy_ii;
reg [7:0] sw_val, hw_val;

initial begin : VERIFY
    // network_done 대기 ($writememh는 이미 각 레이어 완료 시점에 저장됨)
    wait(network_done);
    #(10*CLK_PERIOD);

    // ----------------------------------------------------------
    // CONV00 검증: SW output.hex vs HW hw_ofm.hex (16ch × 256×256)
    // ----------------------------------------------------------
    vfy_err   = 0;
    vfy_total = 16 * 256 * 256;
    vfy_fd_sw = $fopen("CONV00_output.hex",  "r");
    vfy_fd_hw = $fopen("CONV00_hw_ofm.hex",  "r");
    if(vfy_fd_sw == 0 || vfy_fd_hw == 0) begin
        $display("[VERIFY] CONV00: 파일 없음 - 검증 스킵");
    end else begin
        for(vfy_ii = 0; vfy_ii < vfy_total; vfy_ii = vfy_ii + 1) begin
            $fscanf(vfy_fd_sw, "%h\n", sw_val);
            $fscanf(vfy_fd_hw, "%h\n", hw_val);
            if(sw_val !== hw_val) vfy_err = vfy_err + 1;
        end
        $fclose(vfy_fd_sw); $fclose(vfy_fd_hw);
        $display("[VERIFY] CONV00 OFM: total=%0d, errors=%0d -> %s",
                 vfy_total, vfy_err, (vfy_err==0) ? "PASS" : "FAIL");
    end

    // ----------------------------------------------------------
    // CONV02 검증: SW output.hex vs HW hw_ofm.hex (32ch × 128×128)
    // ----------------------------------------------------------
    vfy_err   = 0;
    vfy_total = 32 * 128 * 128;
    vfy_fd_sw = $fopen("CONV02_output.hex",  "r");
    vfy_fd_hw = $fopen("CONV02_hw_ofm.hex",  "r");
    if(vfy_fd_sw == 0 || vfy_fd_hw == 0) begin
        $display("[VERIFY] CONV02: 파일 없음 - 검증 스킵");
    end else begin
        for(vfy_ii = 0; vfy_ii < vfy_total; vfy_ii = vfy_ii + 1) begin
            $fscanf(vfy_fd_sw, "%h\n", sw_val);
            $fscanf(vfy_fd_hw, "%h\n", hw_val);
            if(sw_val !== hw_val) vfy_err = vfy_err + 1;
        end
        $fclose(vfy_fd_sw); $fclose(vfy_fd_hw);
        $display("[VERIFY] CONV02 OFM: total=%0d, errors=%0d -> %s",
                 vfy_total, vfy_err, (vfy_err==0) ? "PASS" : "FAIL");
    end

    // ----------------------------------------------------------
    // CONV04 검증: SW output.hex vs HW hw_ofm.hex (64ch × 64×64)
    // ----------------------------------------------------------
    vfy_err   = 0;
    vfy_total = 64 * 64 * 64;
    vfy_fd_sw = $fopen("CONV04_output.hex",  "r");
    vfy_fd_hw = $fopen("CONV04_hw_ofm.hex",  "r");
    if(vfy_fd_sw == 0 || vfy_fd_hw == 0) begin
        $display("[VERIFY] CONV04: 파일 없음 - 검증 스킵");
    end else begin
        for(vfy_ii = 0; vfy_ii < vfy_total; vfy_ii = vfy_ii + 1) begin
            $fscanf(vfy_fd_sw, "%h\n", sw_val);
            $fscanf(vfy_fd_hw, "%h\n", hw_val);
            if(sw_val !== hw_val) vfy_err = vfy_err + 1;
        end
        $fclose(vfy_fd_sw); $fclose(vfy_fd_hw);
        $display("[VERIFY] CONV04 OFM: total=%0d, errors=%0d -> %s",
                 vfy_total, vfy_err, (vfy_err==0) ? "PASS" : "FAIL");
    end

    $display("[VERIFY] 전체 검증 완료");
end

// ============================================================
// DMA Read 확인용 (입력 이미지 덤프 - 디버그)
// ============================================================
`ifdef CHECK_DMA_WRITE
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

endmodule
