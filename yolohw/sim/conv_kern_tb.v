`timescale 1ns / 1ns

module conv_kern_tb;
`include "../src/define.v"

// Clock
parameter CLK_PERIOD = 10;  // 100MHz
// TO=12: 한 번에 12개 출력채널 병렬 처리
parameter TO          = 12;
// MAC 파이프라인 지연: mul(5) + adder_tree(4) = 9클럭
parameter MAC_LATENCY = 9;

reg clk;
reg rstn;

initial begin
    clk = 1'b1;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

//--------------------------------------------------------------------
// 입력 feature map (전체 Ni채널)
// in_img_all[ch * H * W + row * W + col]
//--------------------------------------------------------------------
reg [7:0]              in_img_all [0:IFM_HEIGHT*IFM_WIDTH*Ni-1];
// 입력 bmp 시각화용 (채널0, 32b)
reg [IFM_WORD_SIZE_32-1:0] in_img [0:IFM_DATA_SIZE_32-1];
// 가중치
reg [IFM_WORD_SIZE_32-1:0] filter [0:WGT_DATA_SIZE-1];

// 누산기 / 출력
reg signed [31:0] accum_reg   [0:TO-1];
reg        [7:0]  conv_out_reg [0:TO-1];
reg               out_vld_reg;

// MAC I/O
reg         vld_i;
reg [127:0] win  [0:TO-1];
reg [127:0] din;
wire [19:0] acc_o [0:TO-1];
wire        vld_o [0:TO-1];

integer i, j, ni_ch;

//--------------------------------------------------------------------
// 파일 로드
//--------------------------------------------------------------------
initial begin: PROC_Load
    for(i = 0; i < IFM_HEIGHT*IFM_WIDTH*Ni; i=i+1) in_img_all[i] = 0;
    $display("Loading ALL channels: %s", IFM_FILE_ALL);
    $readmemh(IFM_FILE_ALL, in_img_all);

    for(i = 0; i < IFM_DATA_SIZE_32; i=i+1) in_img[i] = 0;
    $readmemh(IFM_FILE_32, in_img);

    for(i = 0; i < WGT_DATA_SIZE; i=i+1) filter[i] = 0;
    $display("Loading weights: %s", WGT_FILE);
    $readmemh(WGT_FILE, filter);
end

//--------------------------------------------------------------------
// 메인 컨볼루션 루프 - Ni 누산 포함
//--------------------------------------------------------------------
integer row, col;
reg preload;

initial begin
    rstn        = 1'b0;
    preload     = 1'b0;
    vld_i       = 1'b0;
    out_vld_reg = 1'b0;
    din         = 128'd0;
    for(j=0; j<TO; j=j+1) win[j]         = 128'd0;
    for(j=0; j<TO; j=j+1) accum_reg[j]   = 0;
    for(j=0; j<TO; j=j+1) conv_out_reg[j] = 0;
    row = 0; col = 0;

    #(4*CLK_PERIOD) rstn = 1'b1;
    #(100*CLK_PERIOD) @(posedge clk);

    // 필터 출력 (디버그)
    preload = 1'b1;
    #(100*CLK_PERIOD) @(posedge clk);
    for(j=0; j<TO; j=j+1) begin
        $display("Filter och=%02d:", j);
        for(i=0; i<3; i=i+1)
            $display("  %4d %4d %4d",
                $signed(filter[(j*Fx*Fy*Ni)+(3*i  )][7:0]),
                $signed(filter[(j*Fx*Fy*Ni)+(3*i+1)][7:0]),
                $signed(filter[(j*Fx*Fy*Ni)+(3*i+2)][7:0]));
    end
    #(100*CLK_PERIOD) @(posedge clk);
    preload = 1'b0;

    #(100*CLK_PERIOD);
    $display("[V2] Conv start: %0dx%0d Ni=%0d No=%0d TO=%0d",
             IFM_WIDTH, IFM_HEIGHT, Ni, No, TO);

    // -------------------------------------------------------
    // 픽셀 스캔 루프
    // -------------------------------------------------------
    for(row=0; row<IFM_HEIGHT; row=row+1) begin
        // 행 시작 전 hsync 대기
        #(100*CLK_PERIOD) @(posedge clk);

        for(col=0; col<IFM_WIDTH; col=col+1) begin

            // 누산기 초기화
            for(j=0; j<TO; j=j+1) accum_reg[j] = 0;

            // ------------------------------------------
            // Ni 입력채널 누산 루프
            // ------------------------------------------
            for(ni_ch=0; ni_ch<Ni; ni_ch=ni_ch+1) begin

                // MAC 입력 세팅 + vld_i 어설트
                @(posedge clk) begin
                    vld_i = 1'b1;

                    // din: 현재 ni_ch의 3x3 이웃 (zero padding)
                    din[ 7: 0] = ((row==0)||(col==0))              ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+(col-1)];
                    din[15: 8] =  (row==0)                          ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+ col   ];
                    din[23:16] = ((row==0)||(col==IFM_WIDTH-1))     ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row-1)*IFM_WIDTH+(col+1)];
                    din[31:24] =  (col==0)                          ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+(col-1)];
                    din[39:32] =                                              in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+ col   ];
                    din[47:40] =  (col==IFM_WIDTH-1)                ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+ row   *IFM_WIDTH+(col+1)];
                    din[55:48] = ((row==IFM_HEIGHT-1)||(col==0))    ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+(col-1)];
                    din[63:56] =  (row==IFM_HEIGHT-1)               ? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+ col   ];
                    din[71:64] = ((row==IFM_HEIGHT-1)||(col==IFM_WIDTH-1))? 8'd0 : in_img_all[ni_ch*IFM_HEIGHT*IFM_WIDTH+(row+1)*IFM_WIDTH+(col+1)];
                    din[127:72] = 56'd0;

                    // win: TO개 출력채널의 ni_ch번째 채널 가중치
                    // filter[(출력채널) * Fx*Fy*Ni + ni_ch*9 + [0:8]]
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

                // vld_i 해제
                @(posedge clk) vld_i = 1'b0;

                // MAC 파이프라인 대기 (총 9클럭, 2클럭 이미 경과)
                repeat(MAC_LATENCY-2) @(posedge clk);

                // 누산
                for(j=0; j<TO; j=j+1)
                    accum_reg[j] = accum_reg[j] + $signed(acc_o[j]);

            end // ni_ch

            // ReLU + 역양자화 (>>12)
            for(j=0; j<TO; j=j+1) begin
                if(accum_reg[j] > 0)
                    conv_out_reg[j] = accum_reg[j][19:12];
                else
                    conv_out_reg[j] = 8'd0;
            end

            // bmp writer 유효 펄스
            @(posedge clk) out_vld_reg = 1'b1;
            @(posedge clk) out_vld_reg = 1'b0;

        end // col
    end // row

    $display("[V2] Done: %0dx%0d, TO=%0d channels computed", IFM_WIDTH, IFM_HEIGHT, TO);
    #(100*CLK_PERIOD) @(posedge clk) $stop;
end

//--------------------------------------------------------------------
// MAC 인스턴스 (TO=12개)
//--------------------------------------------------------------------
mac u_mac_00(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 0]),.din(din),.acc_o(acc_o[ 0]),.vld_o(vld_o[ 0]));
mac u_mac_01(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 1]),.din(din),.acc_o(acc_o[ 1]),.vld_o(vld_o[ 1]));
mac u_mac_02(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 2]),.din(din),.acc_o(acc_o[ 2]),.vld_o(vld_o[ 2]));
mac u_mac_03(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 3]),.din(din),.acc_o(acc_o[ 3]),.vld_o(vld_o[ 3]));
mac u_mac_04(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 4]),.din(din),.acc_o(acc_o[ 4]),.vld_o(vld_o[ 4]));
mac u_mac_05(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 5]),.din(din),.acc_o(acc_o[ 5]),.vld_o(vld_o[ 5]));
mac u_mac_06(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 6]),.din(din),.acc_o(acc_o[ 6]),.vld_o(vld_o[ 6]));
mac u_mac_07(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 7]),.din(din),.acc_o(acc_o[ 7]),.vld_o(vld_o[ 7]));
mac u_mac_08(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 8]),.din(din),.acc_o(acc_o[ 8]),.vld_o(vld_o[ 8]));
mac u_mac_09(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[ 9]),.din(din),.acc_o(acc_o[ 9]),.vld_o(vld_o[ 9]));
mac u_mac_10(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[10]),.din(din),.acc_o(acc_o[10]),.vld_o(vld_o[10]));
mac u_mac_11(.clk(clk),.rstn(rstn),.vld_i(vld_i),.win(win[11]),.din(din),.acc_o(acc_o[11]),.vld_o(vld_o[11]));

//--------------------------------------------------------------------
// 출력 bmp 저장 (ch00~ch11)
//--------------------------------------------------------------------
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG00),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out00(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 0]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG01),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out01(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 1]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG02),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out02(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 2]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG03),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out03(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 3]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG04),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out04(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 4]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG05),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out05(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 5]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG06),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out06(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 6]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG07),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out07(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 7]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG08),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out08(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 8]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG09),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out09(.clk(clk),.rstn(rstn),.din(conv_out_reg[ 9]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG10),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out10(.clk(clk),.rstn(rstn),.din(conv_out_reg[10]),.vld(out_vld_reg),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_OUTPUT_IMG11),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_out11(.clk(clk),.rstn(rstn),.din(conv_out_reg[11]),.vld(out_vld_reg),.frame_done());

//--------------------------------------------------------------------
// 입력 이미지 디버그 bmp (채널0)
//--------------------------------------------------------------------
reg dbg_write_image, dbg_write_image_done;
reg [31:0] dbg_pix_idx;
always @(posedge clk, negedge rstn) begin
    if(!rstn) begin dbg_write_image<=0; dbg_write_image_done<=0; dbg_pix_idx<=0; end
    else begin
        if(dbg_write_image) begin
            if(dbg_pix_idx==IFM_DATA_SIZE_32-1) begin dbg_write_image<=0; dbg_write_image_done<=1; dbg_pix_idx<=0; end
            else dbg_pix_idx<=dbg_pix_idx+1;
        end else if(preload) begin dbg_write_image<=1; dbg_write_image_done<=0; dbg_pix_idx<=0; end
    end
end
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG00),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_in00(.clk(clk),.rstn(rstn),.din(in_img[dbg_pix_idx][7:0]),.vld(dbg_write_image),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG01),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_in01(.clk(clk),.rstn(rstn),.din(in_img[dbg_pix_idx][15:8]),.vld(dbg_write_image),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG02),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_in02(.clk(clk),.rstn(rstn),.din(in_img[dbg_pix_idx][23:16]),.vld(dbg_write_image),.frame_done());
bmp_image_writer #(.OUTFILE(CONV_INPUT_IMG03),.WIDTH(IFM_WIDTH),.HEIGHT(IFM_HEIGHT)) u_in03(.clk(clk),.rstn(rstn),.din(in_img[dbg_pix_idx][31:24]),.vld(dbg_write_image),.frame_done());

endmodule
