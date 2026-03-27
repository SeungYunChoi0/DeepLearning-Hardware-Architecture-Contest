`timescale 1ns / 1ns

module conv_kern_tb;
`include "define_CONV04.v" 

parameter CLK_PERIOD = 10; 
parameter TO          = 12; 
parameter MAC_LATENCY = 9;

parameter OUT_PATH_HEX  = "C:/Users/15Z980/Desktop/yun/DeepLearning-Hardware-Architecture-Contest";
parameter CONV_OUTPUT_HEX00 = {OUT_PATH_HEX, "/CONV04_hw_out_ch00.hex"};
parameter CONV_OUTPUT_HEX01 = {OUT_PATH_HEX, "/CONV04_hw_out_ch01.hex"};
parameter CONV_OUTPUT_HEX02 = {OUT_PATH_HEX, "/CONV04_hw_out_ch02.hex"};
parameter CONV_OUTPUT_HEX03 = {OUT_PATH_HEX, "/CONV04_hw_out_ch03.hex"};
parameter BIAS_FILE = "C:/Users/15Z980/Desktop/yun/yolohw/sim/inout_data_sw/log_param/CONV04_param_biases.hex";

reg clk;
reg rstn;

initial begin
    clk = 1'b1;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

//--------------------------------------------------------------------
// 메모리 선언
//--------------------------------------------------------------------
reg [7:0] in_img_all [0:IFM_HEIGHT*IFM_WIDTH*Ni-1]; // 입력 IFM (planar)
reg [IFM_WORD_SIZE_32-1:0] in_img [0:IFM_DATA_SIZE_32-1]; // bmp 시각화용
reg [IFM_WORD_SIZE_32-1:0] filter_pack [0:WGT_DATA_SIZE/4-1]; // TI=4 packed 원본

// ★ 핵심 수정: 언패킹된 8비트 가중치 배열 (yolo_engine.v와 동일 방식)
// WGT_DATA_SIZE = 3*3*32*64 = 18432 개의 INT8 가중치
reg [7:0] wgt [0:WGT_DATA_SIZE-1];

// bias (No=64채널, INT16 signed)
reg signed [15:0] bias [0:63];

reg signed [31:0] accum_reg   [0:TO-1];
reg        [7:0]  conv_out_reg [0:TO-1];
reg               out_vld_reg;

reg         vld_i;
reg [127:0] win  [0:TO-1];
reg [127:0] din;
wire [19:0] acc_o [0:TO-1];
wire        vld_o [0:TO-1];

reg signed [31:0] temp_psum;

integer fp_h0, fp_h1, fp_h2, fp_h3;
integer i, j, ni_ch;
integer row, col;

//--------------------------------------------------------------------
// 데이터 로드 및 언패킹
//--------------------------------------------------------------------
initial begin: PROC_Load
    fp_h0 = $fopen(CONV_OUTPUT_HEX00, "w");
    fp_h1 = $fopen(CONV_OUTPUT_HEX01, "w");
    fp_h2 = $fopen(CONV_OUTPUT_HEX02, "w");
    fp_h3 = $fopen(CONV_OUTPUT_HEX03, "w");

    // IFM 로드
    for(i = 0; i < IFM_HEIGHT*IFM_WIDTH*Ni; i=i+1) in_img_all[i] = 0;
    $display("Loading CONV04 ALL channels IFM...");
    $readmemh(IFM_FILE_ALL, in_img_all);

    for(i = 0; i < IFM_DATA_SIZE_32; i=i+1) in_img[i] = 0;
    $readmemh(IFM_FILE_32, in_img);

    // ★ 가중치: packed hex → 8비트 배열로 언패킹 (yolo_engine.v 방식)
    // filter_pack[n] = {w[4n+3], w[4n+2], w[4n+1], w[4n+0]}
    // packed word 수 = WGT_DATA_SIZE/4 = 18432/4 = 4608
    for(i = 0; i < WGT_DATA_SIZE/4; i=i+1) filter_pack[i] = 0;
    $display("Loading CONV04 weights (packed)...");
    $readmemh(WGT_FILE, filter_pack);

    // 언패킹: 4608개 32비트 word → 18432개 8비트 byte
    for(i = 0; i < WGT_DATA_SIZE/4; i=i+1) begin
        wgt[i*4+0] = filter_pack[i][ 7: 0];
        wgt[i*4+1] = filter_pack[i][15: 8];
        wgt[i*4+2] = filter_pack[i][23:16];
        wgt[i*4+3] = filter_pack[i][31:24];
    end
    $display("Weight unpack done. wgt[0]=%0d wgt[1]=%0d wgt[2]=%0d wgt[3]=%0d",
             $signed(wgt[0]), $signed(wgt[1]), $signed(wgt[2]), $signed(wgt[3]));

    // bias 로드 (64채널, INT16 signed)
    for(i = 0; i < 64; i=i+1) bias[i] = 0;
    $display("Loading CONV04 biases...");
    $readmemh(BIAS_FILE, bias);
    $display("bias[0]=%0d bias[1]=%0d bias[2]=%0d bias[3]=%0d",
             $signed(bias[0]), $signed(bias[1]), $signed(bias[2]), $signed(bias[3]));
end

//--------------------------------------------------------------------
// 메인 시뮬레이션 루프
//--------------------------------------------------------------------
initial begin: PROC_Main
    rstn        = 1'b0;
    vld_i       = 1'b0;
    out_vld_reg = 1'b0;
    din         = 128'd0;
    for(j=0; j<TO; j=j+1) begin
        win[j]          = 128'd0;
        accum_reg[j]    = 0;
        conv_out_reg[j] = 0;
    end
    row = 0; col = 0;

    #(4*CLK_PERIOD) rstn = 1'b1;
    #(100*CLK_PERIOD) @(posedge clk);

    $display("[V2] CONV04 Start: %0dx%0d Ni=%0d No=%0d", IFM_WIDTH, IFM_HEIGHT, Ni, No);

    for(row=0; row<IFM_HEIGHT; row=row+1) begin
        #(10*CLK_PERIOD) @(posedge clk);
        for(col=0; col<IFM_WIDTH; col=col+1) begin

            for(j=0; j<TO; j=j+1) accum_reg[j] = 0;

            for(ni_ch=0; ni_ch<Ni; ni_ch=ni_ch+1) begin
                @(posedge clk) begin: PROC_Input_Pack
                    vld_i = 1'b1;

                    // IFM 3x3 window (zero padding)
                    din[ 7: 0] = ((row==0)||(col==0))               ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+(col-1)];
                    din[15: 8] =  (row==0)                           ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+ col   ];
                    din[23:16] = ((row==0)||(col==IFM_WIDTH-1))      ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+(col+1)];
                    din[31:24] =  (col==0)                           ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+(col-1)];
                    din[39:32] =                                               in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+ col   ];
                    din[47:40] =  (col==IFM_WIDTH-1)                 ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+(col+1)];
                    din[55:48] = ((row==IFM_HEIGHT-1)||(col==0))     ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+(col-1)];
                    din[63:56] =  (row==IFM_HEIGHT-1)                ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+ col   ];
                    din[71:64] = ((row==IFM_HEIGHT-1)||(col==IFM_WIDTH-1))? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+(col+1)];
                    din[127:72] = 56'd0;

                    // ★ 수정된 가중치 인덱싱: 언패킹된 wgt[] 배열 직접 사용
                    // wgt[(출력채널) * Fx*Fy*Ni + ni_ch*9 + [0:8]]
                    for(j=0; j<TO; j=j+1) begin
                        win[j][ 7: 0] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+0];
                        win[j][15: 8] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+1];
                        win[j][23:16] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+2];
                        win[j][31:24] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+3];
                        win[j][39:32] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+4];
                        win[j][47:40] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+5];
                        win[j][55:48] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+6];
                        win[j][63:56] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+7];
                        win[j][71:64] = wgt[(j*Fx*Fy*Ni)+ni_ch*9+8];
                        win[j][127:72] = 56'd0;
                    end
                end

                @(posedge clk) vld_i = 1'b0;
                repeat(MAC_LATENCY-2) @(posedge clk);

                for(j=0; j<TO; j=j+1)
                    accum_reg[j] = accum_reg[j] + $signed({{12{acc_o[j][19]}}, acc_o[j]});
            end

            // ★ ReLU + bias + >>9 + clip
            // CONV04: scale = in_m(32) * w_m(256) / next_m(16) = 512 = 2^9
            // bias[j]: j번 출력채널의 양자화된 bias (INT16)
            for(j=0; j<TO; j=j+1) begin
                temp_psum = accum_reg[j] + {{16{bias[j][15]}}, bias[j]};
                if($signed(temp_psum) > 0) begin
                    if(temp_psum[31:17] != 0)
                        conv_out_reg[j] = 8'hFF;  // overflow → 최대 클리핑
                    else
                        conv_out_reg[j] = temp_psum[16:9]; // >>9
                end else
                    conv_out_reg[j] = 8'd0; // ReLU
            end

            @(posedge clk) begin
                out_vld_reg = 1'b1;
                $fwrite(fp_h0, "%02x\n", conv_out_reg[0]);
                $fwrite(fp_h1, "%02x\n", conv_out_reg[1]);
                $fwrite(fp_h2, "%02x\n", conv_out_reg[2]);
                $fwrite(fp_h3, "%02x\n", conv_out_reg[3]);
            end
            @(posedge clk) out_vld_reg = 1'b0;

        end
    end

    $display("[V2] CONV04 Simulation Done!");
    $fclose(fp_h0); $fclose(fp_h1); $fclose(fp_h2); $fclose(fp_h3);
    #(100*CLK_PERIOD) @(posedge clk) $stop;
end

//--------------------------------------------------------------------
// 모듈 인스턴스
//--------------------------------------------------------------------
generate
    genvar m;
    for(m=0; m<TO; m=m+1) begin: MAC_GEN
        mac u_mac(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[m]),.din(din),.acc_o(acc_o[m]),.vld_o(vld_o[m]));
    end
endgenerate

generate
    genvar b;
    for(b=0; b<4; b=b+1) begin: BMP_OUT_GEN
        bmp_image_writer #(
            .OUTFILE((b==0)?CONV_OUTPUT_IMG00:(b==1)?CONV_OUTPUT_IMG01:(b==2)?CONV_OUTPUT_IMG02:CONV_OUTPUT_IMG03),
            .WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT))
        u_out(.clk(clk),.rstn(rstn),.din(conv_out_reg[b]),.vld(out_vld_reg),.frame_done());
    end
endgenerate

endmodule
