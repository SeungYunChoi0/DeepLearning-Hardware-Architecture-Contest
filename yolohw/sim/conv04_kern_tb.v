`timescale 1ns / 1ns

module conv_kern_tb;
// 설정 파일 로드 (파일명이 일치하는지 확인하세요)
`include "define_CONV04.v" 

// 시뮬레이션 파라미터
parameter CLK_PERIOD = 10; 
parameter TO          = 12; 
parameter MAC_LATENCY = 9;

// HEX 파일 출력 경로 설정
parameter OUT_PATH_HEX = "C:/Users/15Z980/Desktop/yun/DeepLearning-Hardware-Architecture-Contest";
parameter CONV_OUTPUT_HEX00 = {OUT_PATH_HEX, "/CONV04_hw_out_ch00.hex"};
parameter CONV_OUTPUT_HEX01 = {OUT_PATH_HEX, "/CONV04_hw_out_ch01.hex"};
parameter CONV_OUTPUT_HEX02 = {OUT_PATH_HEX, "/CONV04_hw_out_ch02.hex"};
parameter CONV_OUTPUT_HEX03 = {OUT_PATH_HEX, "/CONV04_hw_out_ch03.hex"};

reg clk;
reg rstn;

initial begin
    clk = 1'b1;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

//--------------------------------------------------------------------
// 신호 및 메모리 선언
//--------------------------------------------------------------------
reg [7:0]              in_img_all [0:IFM_HEIGHT*IFM_WIDTH*Ni-1];
reg [IFM_WORD_SIZE_32-1:0] in_img [0:IFM_DATA_SIZE_32-1];
reg [IFM_WORD_SIZE_32-1:0] filter [0:WGT_DATA_SIZE-1];

reg signed [31:0] accum_reg   [0:TO-1];
reg        [7:0]  conv_out_reg [0:TO-1];
reg               out_vld_reg;

reg         vld_i;
reg [127:0] win  [0:TO-1];
reg [127:0] din;
wire [19:0] acc_o [0:TO-1];
wire        vld_o [0:TO-1];

// 문법 에러 방지를 위한 임시 변수들
reg [127:0] tmp_win_val;
reg [31:0]  temp_psum;

// 파일 핸들러
integer fp_h0, fp_h1, fp_h2, fp_h3;

integer i, j, ni_ch;
integer row, col;

//--------------------------------------------------------------------
// 데이터 로드 및 파일 오픈
//--------------------------------------------------------------------
initial begin: PROC_Load
    fp_h0 = $fopen(CONV_OUTPUT_HEX00, "w");
    fp_h1 = $fopen(CONV_OUTPUT_HEX01, "w");
    fp_h2 = $fopen(CONV_OUTPUT_HEX02, "w");
    fp_h3 = $fopen(CONV_OUTPUT_HEX03, "w");

    for(i = 0; i < IFM_HEIGHT*IFM_WIDTH*Ni; i=i+1) in_img_all[i] = 0;
    $display("Loading CONV04 ALL channels...");
    $readmemh(IFM_FILE_ALL, in_img_all);

    for(i = 0; i < IFM_DATA_SIZE_32; i=i+1) in_img[i] = 0;
    $readmemh(IFM_FILE_32, in_img);

    for(i = 0; i < WGT_DATA_SIZE; i=i+1) filter[i] = 0;
    $display("Loading CONV04 weights...");
    $readmemh(WGT_FILE, filter);
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
        win[j]         = 128'd0;
        accum_reg[j]   = 0;
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
                    din[ 7: 0] = ((row==0)||(col==0))              ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+(col-1)];
                    din[15: 8] =  (row==0)                          ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+ col   ];
                    din[23:16] = ((row==0)||(col==IFM_WIDTH-1))     ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+(col+1)];
                    din[31:24] =  (col==0)                          ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+(col-1)];
                    din[39:32] =                                             in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+ col   ];
                    din[47:40] =  (col==IFM_WIDTH-1)                ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+(col+1)];
                    din[55:48] = ((row==IFM_HEIGHT-1)||(col==0))    ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+(col-1)];
                    din[63:56] =  (row==IFM_HEIGHT-1)               ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+ col   ];
                    din[71:64] = ((row==IFM_HEIGHT-1)||(col==IFM_WIDTH-1))? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+(col+1)];
                    din[127:72] = 56'd0;

                    for(j=0; j<TO; j=j+1) begin
                        // 임시 변수 tmp_win_val에 먼저 값을 채운 뒤 배열 win[j]에 대입
                        tmp_win_val[ 7: 0] = filter[(j*Fx*Fy*Ni)+ni_ch*9+0][7:0];
                        tmp_win_val[15: 8] = filter[(j*Fx*Fy*Ni)+ni_ch*9+1][7:0];
                        tmp_win_val[23:16] = filter[(j*Fx*Fy*Ni)+ni_ch*9+2][7:0];
                        tmp_win_val[31:24] = filter[(j*Fx*Fy*Ni)+ni_ch*9+3][7:0];
                        tmp_win_val[39:32] = filter[(j*Fx*Fy*Ni)+ni_ch*9+4][7:0];
                        tmp_win_val[47:40] = filter[(j*Fx*Fy*Ni)+ni_ch*9+5][7:0];
                        tmp_win_val[55:48] = filter[(j*Fx*Fy*Ni)+ni_ch*9+6][7:0];
                        tmp_win_val[63:56] = filter[(j*Fx*Fy*Ni)+ni_ch*9+7][7:0];
                        tmp_win_val[71:64] = filter[(j*Fx*Fy*Ni)+ni_ch*9+8][7:0];
                        tmp_win_val[127:72] = 56'd0;
                        win[j] = tmp_win_val;
                    end
                end
                @(posedge clk) vld_i = 1'b0;
                repeat(MAC_LATENCY-2) @(posedge clk);
                for(j=0; j<TO; j=j+1)
                    accum_reg[j] = accum_reg[j] + $signed(acc_o[j]);
            end 

            // ReLU 및 역양자화 (>> 9)
            for(j=0; j<TO; j=j+1) begin
                temp_psum = accum_reg[j];
                if($signed(temp_psum) > 0) begin
                    if(temp_psum[31:17] != 0) 
                        conv_out_reg[j] = 8'hFF;
                    else
                        conv_out_reg[j] = temp_psum[16:9]; 
                end else
                    conv_out_reg[j] = 8'd0;
            end

            // HEX 파일 기록 및 유효 신호 출력
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
        bmp_image_writer #(.OUTFILE( (b==0)?CONV_OUTPUT_IMG00:(b==1)?CONV_OUTPUT_IMG01:(b==2)?CONV_OUTPUT_IMG02:CONV_OUTPUT_IMG03 ),
                          .WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) 
        u_out(.clk(clk),.rstn(rstn),.din(conv_out_reg[b]),.vld(out_vld_reg),.frame_done());
    end
endgenerate

endmodule