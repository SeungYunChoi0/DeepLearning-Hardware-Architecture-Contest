`timescale 1ns / 1ns

module conv_kern_tb;
// 파일명을 사용자가 수정한 이름으로 변경했습니다. 
// 만약 src 폴더 안에 있다면 "../src/define_CONV04.v"로 경로를 맞춰주세요.
`include "define_CONV04.v" 

// Clock
parameter CLK_PERIOD = 10; 
parameter TO          = 12; // 64개 출력 채널 중 12개 병렬 처리
parameter MAC_LATENCY = 9;

reg clk;
reg rstn;

initial begin
    clk = 1'b1;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

//--------------------------------------------------------------------
// 메모리 선언
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

integer i, j, ni_ch;

//--------------------------------------------------------------------
// 데이터 로드
//--------------------------------------------------------------------
initial begin: PROC_Load
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
// 메인 시뮬레이션 로직
//--------------------------------------------------------------------
integer row, col;
reg preload;

initial begin
    rstn        = 1'b0;
    preload     = 1'b0;
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
                @(posedge clk) begin
                    vld_i = 1'b1;
                    // Input packing (3x3 window with zero padding)
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
                        win[j][ 7: 0] = filter[(j*Fx*Fy*Ni)+ni_ch*9+0][7:0];
                        win[j][15: 8] = filter[(j*Fx*Fy*Ni)+ni_ch*9+1][7:0];
                        win[j][23:16] = filter[(j*Fx*Fy*Ni)+ni_ch*9+2][7:0];
                        win[j][31:24] = filter[(j*Fx*Fy*Ni)+ni_ch*9+3][7:0];
                        win[j][39:32] = filter[(j*Fx*Fy*Ni)+ni_ch*9+4][7:0];
                        win[j][47:40] = filter[(j*Fx*Fy*Ni)+ni_ch*9+5][7:0];
                        win[j][55:48] = filter[(j*Fx*Fy*Ni)+ni_ch*9+6][7:0];
                        win[j][63:56] = filter[(j*Fx*Fy*Ni)+ni_ch*9+7][7:0];
                        win[j][71:64] = filter[(j*Fx*Fy*Ni)+ni_ch*9+8][7:0];
                        win[j][127:72] = 56'd0;
                    end
                end
                @(posedge clk) vld_i = 1'b0;
                repeat(MAC_LATENCY-2) @(posedge clk);
                for(j=0; j<TO; j=j+1)
                    accum_reg[j] = accum_reg[j] + $signed(acc_o[j]);
            end 

            for(j=0; j<TO; j=j+1) begin
                if(accum_reg[j] > 0)
                    conv_out_reg[j] = accum_reg[j][19:12];
                else
                    conv_out_reg[j] = 8'd0;
            end

            @(posedge clk) out_vld_reg = 1'b1;
            @(posedge clk) out_vld_reg = 1'b0;
        end
    end

    $display("[V2] CONV04 Simulation Done!");
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