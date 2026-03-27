`timescale 1ns / 1ns

module conv_kern_tb;
`include "../src/define_CONV04.v"

// Clock
parameter CLK_PERIOD = 10; // 100MHz
// TO=12: 한 번에 12개 출력채널 병렬 처리 (No=64 중 처음 12개 검증) [cite: 2]
parameter TO          = 12;
// MAC 파이프라인 지연: mul(5) + adder_tree(4) = 9클럭 [cite: 3]
parameter MAC_LATENCY = 9;

reg clk;
reg rstn;

initial begin
    clk = 1'b1;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

//--------------------------------------------------------------------
// 입력 feature map (전체 Ni=32채널) [cite: 5, 12]
//--------------------------------------------------------------------
reg [7:0]              in_img_all [0:IFM_HEIGHT*IFM_WIDTH*Ni-1];
// 시각화용 (32b 패킹 데이터) [cite: 6]
reg [IFM_WORD_SIZE_32-1:0] in_img [0:IFM_DATA_SIZE_32-1];
// 가중치 [cite: 6]
reg [IFM_WORD_SIZE_32-1:0] filter [0:WGT_DATA_SIZE-1];

// 누산기 / 출력 [cite: 7-8]
reg signed [31:0] accum_reg   [0:TO-1];
reg        [7:0]  conv_out_reg [0:TO-1];
reg               out_vld_reg;

// MAC I/O [cite: 10]
reg         vld_i;
reg [127:0] win  [0:TO-1];
reg [127:0] din;
wire [19:0] acc_o [0:TO-1];
wire        vld_o [0:TO-1];

integer i, j, ni_ch;

//--------------------------------------------------------------------
// 파일 로드 (define.v에 정의된 CONV04 경로 사용) [cite: 12-14]
//--------------------------------------------------------------------
initial begin: PROC_Load
    for(i = 0; i < IFM_HEIGHT*IFM_WIDTH*Ni; i=i+1) in_img_all[i] = 0;
    $display("Loading CONV04 ALL channels: %s", IFM_FILE_ALL); [cite: 13]
    $readmemh(IFM_FILE_ALL, in_img_all);

    for(i = 0; i < IFM_DATA_SIZE_32; i=i+1) in_img[i] = 0;
    $readmemh(IFM_FILE_32, in_img);

    for(i = 0; i < WGT_DATA_SIZE; i=i+1) filter[i] = 0;
    $display("Loading CONV04 weights: %s", WGT_FILE); [cite: 14]
    $readmemh(WGT_FILE, filter);
end

//--------------------------------------------------------------------
// 메인 컨볼루션 루프 - Ni(32) 누산 포함 [cite: 15-20]
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

    // 필터 확인 (디버그) [cite: 22-24]
    preload = 1'b1;
    #(100*CLK_PERIOD) @(posedge clk);
    for(j=0; j<TO; j=j+1) begin
        $display("CONV04 Filter och=%02d:", j);
        for(i=0; i<3; i=i+1)
            $display("  %4d %4d %4d",
                $signed(filter[(j*Fx*Fy*Ni)+(3*i  )][7:0]),
                $signed(filter[(j*Fx*Fy*Ni)+(3*i+1)][7:0]),
                $signed(filter[(j*Fx*Fy*Ni)+(3*i+2)][7:0]));
    end
    preload = 1'b0;

    #(100*CLK_PERIOD);
    $display("[V2] CONV04 Start: %0dx%0d Ni=%0d No=%0d", IFM_WIDTH, IFM_HEIGHT, Ni, No); [cite: 25]

    // 픽셀 스캔 루프 (64x64) 
    for(row=0; row<IFM_HEIGHT; row=row+1) begin
        #(100*CLK_PERIOD) @(posedge clk);
        for(col=0; col<IFM_WIDTH; col=col+1) begin

            // 누산기 초기화 (새로운 픽셀 시작) [cite: 27]
            for(j=0; j<TO; j=j+1) accum_reg[j] = 0;

            // Ni(32) 입력채널 누산 루프 [cite: 28]
            for(ni_ch=0; ni_ch<Ni; ni_ch=ni_ch+1) begin
                @(posedge clk) begin
                    vld_i = 1'b1;
                    // din: 현재 ni_ch의 3x3 데이터 (Zero padding 처리) 
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

                    // win: 출력채널별 ni_ch번째 채널 가중치 
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
                repeat(MAC_LATENCY-2) @(posedge clk); [cite: 41]
                // 누산 (Ni=32번 반복) [cite: 42]
                for(j=0; j<TO; j=j+1) accum_reg[j] = accum_reg[j] + $signed(acc_o[j]);
            end // ni_ch

            // ReLU + 역양자화 (CONV04 스케일에 맞춰 조정 필요 시 수정) [cite: 43-44]
            for(j=0; j<TO; j=j+1) begin
                if(accum_reg[j] > 0)
                    conv_out_reg[j] = accum_reg[j][19:12]; // [V2] 기본 12비트 시프트
                else
                    conv_out_reg[j] = 8'd0;
            end

            // 결과 저장 유효 신호 [cite: 45-46]
            @(posedge clk) out_vld_reg = 1'b1;
            @(posedge clk) out_vld_reg = 1'b0;
        end // col
    end // row

    $display("[V2] CONV04 Done: %0dx%0d, TO=%0d computed", IFM_WIDTH, IFM_HEIGHT, TO); [cite: 46]
    #(100*CLK_PERIOD) @(posedge clk) $stop;
end

// MAC 인스턴스 (TO=12개 유지) [cite: 47-50]
generate
    genvar m;
    for(m=0; m<TO; m=m+1) begin: MAC_GEN
        mac u_mac(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[m]),.din(din),.acc_o(acc_o[m]),.vld_o(vld_o[m]));
    end
endgenerate

// 출력 bmp 저장 (define.v의 CONV04 경로 사용) [cite: 50-52]
generate
    genvar b;
    for(b=0; b<4; b=b+1) begin: BMP_OUT_GEN // 예시로 4개 채널만 파일 저장
        bmp_image_writer #(.OUTFILE( (b==0)?CONV_OUTPUT_IMG00:(b==1)?CONV_OUTPUT_IMG01:(b==2)?CONV_OUTPUT_IMG02:CONV_OUTPUT_IMG03 ),
                          .WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) 
        u_out(.clk(clk),.rstn(rstn),.din(conv_out_reg[b]),.vld(out_vld_reg),.frame_done());
    end
endgenerate

endmodule