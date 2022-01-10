
`define ML 16
`define W1 2841
`define W2 2676
`define W3 2408
`define W5 1609
`define W6 1108
`define W7 565

module idctrow(input [`ML-1:0] i0, input [`ML-1:0] i1, input [`ML-1:0] i2, input [`ML-1:0] i3,
               input [`ML-1:0] i4, input [`ML-1:0] i5, input [`ML-1:0] i6, input [`ML-1:0] i7,
               output [`ML-1:0] b0, output [`ML-1:0] b1, output [`ML-1:0] b2, output [`ML-1:0] b3,
               output [`ML-1:0] b4, output [`ML-1:0] b5, output [`ML-1:0] b6, output [`ML-1:0] b7);

wire signed [`ML*2-1:0] x [7:0][3:0];
wire return_zero;

wire [`ML-1:0] b0;
wire [`ML-1:0] b1;
wire [`ML-1:0] b2;
wire [`ML-1:0] b3;
wire [`ML-1:0] b4;
wire [`ML-1:0] b5;
wire [`ML-1:0] b6;
wire [`ML-1:0] b7;

// zeroth stage
assign x[0][0] = ($signed(i0) << 11) + 128;
assign x[1][0] = $signed(i4) << 11;
assign x[2][0] = $signed(i6);
assign x[3][0] = $signed(i2);
assign x[4][0] = $signed(i1);
assign x[5][0] = $signed(i7);
assign x[6][0] = $signed(i5);
assign x[7][0] = $signed(i3);

assign br = !(x[1][0] | x[2][0] | x[3][0] | x[4][0] | x[5][0] | x[6][0] | x[7][0]);

// first stage
wire signed [`ML*2-1:0] tmp0;
wire signed [`ML*2-1:0] tmp1;
assign x[0][1] = x[0][0];
assign x[1][1] = x[1][0];
assign x[2][1] = x[2][0];
assign x[3][1] = x[3][0];
assign tmp0    = `W7 * (x[4][0] + x[5][0]);
assign x[4][1] = tmp0 + (`W1 - `W7) * x[4][0];
assign x[5][1] = tmp0 - (`W1 + `W7) * x[5][0];
assign tmp1    = `W3 * (x[6][0] + x[7][0]);
assign x[6][1] = tmp1 - (`W3 - `W5) * x[6][0];
assign x[7][1] = tmp1 - (`W3 + `W5) * x[7][0];

// second stage
wire signed [`ML*2-1:0] tmp2;
wire signed [`ML*2-1:0] tmp3;
assign x[0][2] = x[0][1] - x[1][1];
assign x[1][2] = x[4][1] + x[6][1];
assign tmp2 = `W6 * (x[3][1] + x[2][1]);
assign x[2][2] = tmp2 - (`W2 + `W6) * x[2][1];
assign x[3][2] = tmp2 + (`W2 - `W6) * x[3][1];
assign x[4][2] = x[4][1] - x[6][1];
assign x[5][2] = x[5][1] - x[7][1];
assign x[6][2] = x[5][1] + x[7][1];
assign x[7][2] = x[7][1];
assign tmp3 = x[0][1] + x[1][1];

// third stage
wire signed [`ML*2-1:0] tmp4;
assign x[0][3] = x[0][2] - x[2][2];
assign x[1][3] = x[1][2];
assign x[2][3] = (181 * (x[4][2] + x[5][2]) + 128) >>> 8;
assign x[3][3] = x[0][2] + x[2][2];
assign x[4][3] = (181 * (x[4][2] - x[5][2]) + 128) >>> 8;
assign x[5][3] = x[5][2];
assign x[6][3] = x[6][2];
assign x[7][3] = tmp3 +  x[3][2];
assign tmp4    = tmp3 - x[3][2];

// fourth stage
wire signed [`ML*2-1:0] tmp5;
assign tmp5 = $signed(i0) << 3;
assign b0 = br ? tmp5[`ML-1:0] : (x[7][3] + x[1][3]) >>> 8;
assign b1 = br ? tmp5[`ML-1:0] : (x[3][3] + x[2][3]) >>> 8;
assign b2 = br ? tmp5[`ML-1:0] : (x[0][3] + x[4][3]) >>> 8;
assign b3 = br ? tmp5[`ML-1:0] : (   tmp4 + x[6][3]) >>> 8;
assign b4 = br ? tmp5[`ML-1:0] : (   tmp4 - x[6][3]) >>> 8;
assign b5 = br ? tmp5[`ML-1:0] : (x[0][3] - x[4][3]) >>> 8;
assign b6 = br ? tmp5[`ML-1:0] : (x[3][3] - x[2][3]) >>> 8;
assign b7 = br ? tmp5[`ML-1:0] : (x[7][3] - x[1][3]) >>> 8;

endmodule // idctrow

module idctcol(input [`ML-1:0] i0, input [`ML-1:0] i1, input [`ML-1:0] i2, input [`ML-1:0] i3,
               input [`ML-1:0] i4, input [`ML-1:0] i5, input [`ML-1:0] i6, input [`ML-1:0] i7,
               output [`ML-1:0] b0, output [`ML-1:0] b1, output [`ML-1:0] b2, output [`ML-1:0] b3,
               output [`ML-1:0] b4, output [`ML-1:0] b5, output [`ML-1:0] b6, output [`ML-1:0] b7);

wire signed [`ML*2-1:0] x [7:0][3:0];
wire return_zero;

wire [`ML-1:0] b0;
wire [`ML-1:0] b1;
wire [`ML-1:0] b2;
wire [`ML-1:0] b3;
wire [`ML-1:0] b4;
wire [`ML-1:0] b5;
wire [`ML-1:0] b6;
wire [`ML-1:0] b7;

// zeroth stage
assign x[0][0] = ($signed(i0) << 8) + 8192;
assign x[1][0] = $signed(i4) << 8;
assign x[2][0] = $signed(i6);
assign x[3][0] = $signed(i2);
assign x[4][0] = $signed(i1);
assign x[5][0] = $signed(i7);
assign x[6][0] = $signed(i5);
assign x[7][0] = $signed(i3);

assign br = !(x[1][0] | x[2][0] | x[3][0] | x[4][0] | x[5][0] | x[6][0] | x[7][0]);

// first stage
wire signed [`ML*2-1:0] tmp0;
wire signed [`ML*2-1:0] tmp1;
assign x[0][1] = x[0][0];
assign x[1][1] = x[1][0];
assign x[2][1] = x[2][0];
assign x[3][1] = x[3][0];
assign tmp0    = `W7 * (x[4][0] + x[5][0]) + 4;
assign x[4][1] = (tmp0 + (`W1 - `W7) * x[4][0]) >>> 3;
assign x[5][1] = (tmp0 - (`W1 + `W7) * x[5][0]) >>> 3;
assign tmp1    = `W3 * (x[6][0] + x[7][0]) + 4;
assign x[6][1] = (tmp1 - (`W3 - `W5) * x[6][0]) >>> 3;
assign x[7][1] = (tmp1 - (`W3 + `W5) * x[7][0]) >>> 3;

// second stage
wire signed [`ML*2-1:0] tmp2;
wire signed [`ML*2-1:0] tmp3;
assign x[0][2] = x[0][1] - x[1][1];
assign x[1][2] = x[4][1] + x[6][1];
assign tmp2 = `W6 * (x[3][1] + x[2][1]) + 4;
assign x[2][2] = (tmp2 - (`W2 + `W6) * x[2][1]) >>> 3;
assign x[3][2] = (tmp2 + (`W2 - `W6) * x[3][1]) >>> 3;
assign x[4][2] = x[4][1] - x[6][1];
assign x[5][2] = x[5][1] - x[7][1];
assign x[6][2] = x[5][1] + x[7][1];
assign x[7][2] = x[7][1];
assign tmp3 = x[0][1] + x[1][1];

// third stage
wire signed [`ML*2-1:0] tmp4;
assign x[0][3] = x[0][2] - x[2][2];
assign x[1][3] = x[1][2];
assign x[2][3] = (181 * (x[4][2] + x[5][2]) + 128) >>> 8;
assign x[3][3] = x[0][2] + x[2][2];
assign x[4][3] = (181 * (x[4][2] - x[5][2]) + 128) >>> 8;
assign x[5][3] = x[5][2];
assign x[6][3] = x[6][2];
assign x[7][3] = tmp3 + x[3][2];
assign tmp4    = tmp3 - x[3][2];

// fourth stage
wire signed [`ML-1:0] tmp5 = iclp16(($signed(i0) + 32) >>> 6);
assign b0 = br ? tmp5 : iclp16((x[7][3] + x[1][3]) >>> 14);
assign b1 = br ? tmp5 : iclp16((x[3][3] + x[2][3]) >>> 14);
assign b2 = br ? tmp5 : iclp16((x[0][3] + x[4][3]) >>> 14);
assign b3 = br ? tmp5 : iclp16((   tmp4 + x[6][3]) >>> 14);
assign b4 = br ? tmp5 : iclp16((   tmp4 - x[6][3]) >>> 14);
assign b5 = br ? tmp5 : iclp16((x[0][3] - x[4][3]) >>> 14);
assign b6 = br ? tmp5 : iclp16((x[3][3] - x[2][3]) >>> 14);
assign b7 = br ? tmp5 : iclp16((x[7][3] - x[1][3]) >>> 14);

function [15:0] iclp16;
  input [15:0] in;
  begin
    if (in[15] == 1 && in[14:8] != 7'h7F)
      iclp16 = 8'h80;
    else if (in[15] == 0 && in [14:8] != 0)
      iclp16 = 8'h7F;
    else
      iclp16 = in;
  end
endfunction

endmodule // idctcol


module Fast_IDCT(input [`ML*8*8-1:0] in, output [`ML*8*8-1:0] out);

wire [`ML*8*8-1:0] ws;
wire [`ML*8*8-1:0] out;

genvar i;
generate for (i=1; i<8*8; i=i+8) begin
    //b[0]         b[1]         b[2]            b[3]            b[4]            b[5]            b[6]            b[7]
    //b[8]
    //                                                                                                          b[63]
    //1*x-1:0*x	   2*x-1:1*x	3*x-1:2*x	4*x-1:3*x	5*x-1:4*x	6*x-1:5*x	7*x-1:6*x	8*x-1:7*x
    //9*x-1:8*x
    //                                                                                                         64*x-1:63*x
    idctrow ir(in[(i+0)*`ML-1 : (i-1)*`ML],
               in[(i+1)*`ML-1 : (i+0)*`ML],
               in[(i+2)*`ML-1 : (i+1)*`ML],
               in[(i+3)*`ML-1 : (i+2)*`ML],
               in[(i+4)*`ML-1 : (i+3)*`ML],
               in[(i+5)*`ML-1 : (i+4)*`ML],
               in[(i+6)*`ML-1 : (i+5)*`ML],
               in[(i+7)*`ML-1 : (i+6)*`ML],
               ws[(i+0)*`ML-1 : (i-1)*`ML],
               ws[(i+1)*`ML-1 : (i+0)*`ML],
               ws[(i+2)*`ML-1 : (i+1)*`ML],
               ws[(i+3)*`ML-1 : (i+2)*`ML],
               ws[(i+4)*`ML-1 : (i+3)*`ML],
               ws[(i+5)*`ML-1 : (i+4)*`ML],
               ws[(i+6)*`ML-1 : (i+5)*`ML],
               ws[(i+7)*`ML-1 : (i+6)*`ML]);
    //b[0]	b[8]	b[16]					b[56]
    //b[1]
    //                                                          b[63]
    //(1+8*0)*x-1:(0+8*0)*x (1+8*1)*x-1:(0+8*1)*x (1+8*2)*x-1:(0+8*2)*x     (1+8*7)*x-1:(0+8*7)*x
    //(2+8*0)*x-1:(1+8*0)*x
    //                                                                      (8+8*7)*x-1:(7+8*7)*x
end
endgenerate
generate for (i=1; i<=8; i=i+1) begin
    idctcol ic(ws[(i+8*0)*`ML-1 : (i-1+8*0)*`ML],
               ws[(i+8*1)*`ML-1 : (i-1+8*1)*`ML],
               ws[(i+8*2)*`ML-1 : (i-1+8*2)*`ML],
               ws[(i+8*3)*`ML-1 : (i-1+8*3)*`ML],
               ws[(i+8*4)*`ML-1 : (i-1+8*4)*`ML],
               ws[(i+8*5)*`ML-1 : (i-1+8*5)*`ML],
               ws[(i+8*6)*`ML-1 : (i-1+8*6)*`ML],
               ws[(i+8*7)*`ML-1 : (i-1+8*7)*`ML],
               out[(i+8*0)*`ML-1 : (i-1+8*0)*`ML],
               out[(i+8*1)*`ML-1 : (i-1+8*1)*`ML],
               out[(i+8*2)*`ML-1 : (i-1+8*2)*`ML],
               out[(i+8*3)*`ML-1 : (i-1+8*3)*`ML],
               out[(i+8*4)*`ML-1 : (i-1+8*4)*`ML],
               out[(i+8*5)*`ML-1 : (i-1+8*5)*`ML],
               out[(i+8*6)*`ML-1 : (i-1+8*6)*`ML],
               out[(i+8*7)*`ML-1 : (i-1+8*7)*`ML]);
end
endgenerate
endmodule // Fast_IDCT

module top ();
integer i;
integer j;
reg signed [`ML-1:0] b [63:0];
wire [`ML*8*8-1:0] out;
reg [`ML*8*8-1:0] out_reg;
reg [`ML-1:0] buffer;


reg signed [`ML-1:0] exres1 [63:0];
reg signed [`ML-1:0] exres2 [63:0];
reg signed [`ML-1:0] exres3 [63:0];
reg signed [`ML-1:0] exres4 [63:0];


Fast_IDCT idct({b[63], b[62], b[61], b[60], b[59], b[58], b[57], b[56],
                b[55], b[54], b[53], b[52], b[51], b[50], b[49], b[48],
                b[47], b[46], b[45], b[44], b[43], b[42], b[41], b[40],
                b[39], b[38], b[37], b[36], b[35], b[34], b[33], b[32],
                b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                b[23], b[22], b[21], b[20], b[19], b[18], b[17], b[16],
                b[15], b[14], b[13], b[12], b[11], b[10], b[09], b[08],
                b[07], b[06], b[05], b[04], b[03], b[02], b[01], b[00]}, out);
initial begin

  //Expected result for test 1.
  for (i = 0; i < 64; i = i + 1) begin
    exres1[i] <= 3;
  end
  for (i = 0; i < 8; i = i + 1) begin
    exres1[i * 8] <= 2;
  end
  
  //Expected result for test 2.
  for (i = 0; i < 4; i = i + 1) begin
    exres2[i * 8] <= 1;
    exres2[i * 8 + 1] <= 1;
    exres2[i * 8 + 2] <= 1;
    exres2[i * 8 + 3] <= 1;
    exres2[i * 8 + 4] <= 2;
    exres2[i * 8 + 5] <= 2;
    exres2[i * 8 + 6] <= 2;
    exres2[i * 8 + 7] <= 2;
  end
  
  exres2[23] <= 3;
  exres2[30] <= 3;
  exres2[31] <= 3;
  
  exres2[32] <= 0;
  exres2[33] <= 1;
  exres2[34] <= 1;
  exres2[35] <= 1;
  exres2[36] <= 2;
  exres2[37] <= 2;
  exres2[38] <= 3;
  exres2[39] <= 3;
  
  exres2[40] <= 0;
  exres2[41] <= 0;
  exres2[42] <= 1;
  exres2[43] <= 1;
  exres2[44] <= 2;
  exres2[45] <= 2;
  exres2[46] <= 3;
  exres2[47] <= 3;
  
  exres2[48] <= 0;
  exres2[49] <= 0;
  exres2[50] <= 1;
  exres2[51] <= 1;
  exres2[52] <= 2;
  exres2[53] <= 3;
  exres2[54] <= 3;
  exres2[55] <= 3;
  
  exres2[56] <= 0;
  exres2[57] <= 0;
  exres2[58] <= 1;
  exres2[59] <= 1;
  exres2[60] <= 2;
  exres2[61] <= 3;
  exres2[62] <= 3;
  exres2[63] <= 3;
  
  //Expected result for test 3.
  exres3[0] <= -24;
  exres3[1] <= -23;
  exres3[2] <= -21;
  exres3[3] <= -21;
  exres3[4] <= -21;
  exres3[5] <= -21;
  exres3[6] <= -21;
  exres3[7] <= -20;
  
  exres3[8] <= -24;
  exres3[9] <= -22;
  exres3[10] <= -21;
  exres3[11] <= -20;
  exres3[12] <= -21;
  exres3[13] <= -21;
  exres3[14] <= -21;
  exres3[15] <= -20;
  
  exres3[16] <= -23;
  exres3[17] <= -22;
  exres3[18] <= -21;
  exres3[19] <= -20;
  exres3[20] <= -20;
  exres3[21] <= -21;
  exres3[22] <= -20;
  exres3[23] <= -20;
  
  exres3[24] <= -23;
  exres3[25] <= -22;
  exres3[26] <= -20;
  exres3[27] <= -20;
  exres3[28] <= -20;
  exres3[29] <= -20;
  exres3[30] <= -20;
  exres3[31] <= -19;
  
  exres3[32] <= -23;
  exres3[33] <= -22;
  exres3[34] <= -20;
  exres3[35] <= -20;
  exres3[36] <= -20;
  exres3[37] <= -20;
  exres3[38] <= -20;
  exres3[39] <= -19;
  
  exres3[40] <= -23;
  exres3[41] <= -22;
  exres3[42] <= -20;
  exres3[43] <= -20;
  exres3[44] <= -20;
  exres3[45] <= -20;
  exres3[46] <= -20;
  exres3[47] <= -19;
  
  exres3[48] <= -23;
  exres3[49] <= -22;
  exres3[50] <= -20;
  exres3[51] <= -20;
  exres3[52] <= -20;
  exres3[53] <= -20;
  exres3[54] <= -20;
  exres3[55] <= -19;
  
  exres3[56] <= -23;
  exres3[57] <= -22;
  exres3[58] <= -20;
  exres3[59] <= -20;
  exres3[60] <= -20;
  exres3[61] <= -20;
  exres3[62] <= -20;
  exres3[63] <= -20;
  
  //Exrected result for test 4.
  exres4[0] <= 21;
  exres4[1] <= -10;
  exres4[2] <= -26;
  exres4[3] <= -61;
  exres4[4] <= -43;
  exres4[5] <= -17;
  exres4[6] <= -22;
  exres4[7] <= -8;
  
  exres4[8] <= 5;
  exres4[9] <= -28;
  exres4[10] <= -47;
  exres4[11] <= -73;
  exres4[12] <= -11;
  exres4[13] <= -14;
  exres4[14] <= -24;
  exres4[15] <= -17;
  
  exres4[16] <= -14;
  exres4[17] <= -31;
  exres4[18] <= -61;
  exres4[19] <= -45;
  exres4[20] <= -5;
  exres4[21] <= -18;
  exres4[22] <= -22;
  exres4[23] <= -34;
  
  exres4[24] <= -23;
  exres4[25] <= -36;
  exres4[26] <= -49;
  exres4[27] <= -32;
  exres4[28] <= -12;
  exres4[29] <= -33;
  exres4[30] <= -33;
  exres4[31] <= -35;
  
  exres4[32] <= -30;
  exres4[33] <= -39;
  exres4[34] <= -53;
  exres4[35] <= -8;
  exres4[36] <= -19;
  exres4[37] <= -31;
  exres4[38] <= -43;
  exres4[39] <= -42;
  
  exres4[40] <= -41;
  exres4[41] <= -43;
  exres4[42] <= -50;
  exres4[43] <= -4;
  exres4[44] <= -15;
  exres4[45] <= -33;
  exres4[46] <= -44;
  exres4[47] <= -66;
  
  exres4[48] <= -40;
  exres4[49] <= -38;
  exres4[50] <= -21;
  exres4[51] <= -14;
  exres4[52] <= -17;
  exres4[53] <= -26;
  exres4[54] <= -46;
  exres4[55] <= -52;
  
  exres4[56] <= -44;
  exres4[57] <= -47;
  exres4[58] <= -9;
  exres4[59] <= -12;
  exres4[60] <= -30;
  exres4[61] <= -33;
  exres4[62] <= -38;
  exres4[63] <= -37;
  
  
  $dumpfile("test.vcd");
  $dumpvars(6, top);
  
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  
  b[0] <= 23;
  b[1] <= -1;
  b[2] <= -2;
  
  #10;
  out_reg <= out;
  
  #10;
  for (i = 0; i < 64; i = i + 1) begin
    for (j = 0; j < 16; j = j + 1) begin
      buffer[`ML - 1 - j] = out_reg[((i + 1) * 16 - 1) - j];
    end
    if (buffer != exres1[i]) begin
      $display("Unexpected result in test 1: expected %d got %d!", exres1[i], buffer);
      $finish;
    end
  end
  $display("TEST 1 PASSED!");
  #10;
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  b[0] <= 13;
  b[1] <= -7;
  b[9] <= 2;
  
  #10;
  out_reg <= out;
  
  #10;
  for (i = 0; i < 64; i = i + 1) begin
    for (j = 0; j < 16; j = j + 1) begin
      buffer[`ML - 1 - j] = out_reg[((i + 1) * 16 - 1) - j];
    end
    if (buffer != exres2[i]) begin
      $display("Unexpected result in test 2: expected %d got %d!", exres2[i], buffer);
      $finish;
    end
  end
  $display("TEST 2 PASSED!");
  #10;

  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  b[0] <= -166;
  b[1] <= -7;
  b[2] <= -4;
  b[3] <= -4;
  b[8] <= -2;
  b[16] <= -2;

  #10;
  out_reg <= out;

  #10;
  for (i = 0; i < 64; i = i + 1) begin
    for (j = 0; j < 16; j = j + 1) begin
      buffer[`ML - 1 - j] = out_reg[((i + 1) * 16 - 1) - j];
    end
    if (buffer != exres3[i]) begin
      $display("Unexpected result in test 3: expected %d got %d!", exres3[i], buffer);
      $finish;
    end
  end
  $display("TEST 3 PASSED!");
  #10;
  
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  
  b[0] <= -240;
  b[1] <= 8;
  b[2] <= -11;
  b[3] <= 47;
  b[4] <= 26;
  b[5] <= -6;
  b[7] <= 5;
  
  b[8] <= 28;
  b[9] <= -6;
  b[10] <= 85;
  b[11] <= 44;
  b[12] <= -4;
  b[13] <= -25;
  b[14] <= 5;
  b[15] <= 16;
  
  b[16] <= 21;
  b[17] <= 8;
  b[18] <= 32;
  b[19] <= -16;
  b[20] <= -24;
  b[22] <= 30;
  b[23] <= 12;
  
  b[24] <= -2;
  b[25] <= 18;
  b[27] <= -2;
  b[29] <= 7;
  b[31] <= -15;
  
  b[32] <= 7;
  b[33] <= 4;
  b[34] <= 15;
  b[35] <= -24;
  b[37] <= 9;
  b[38] <= 8;
  b[39] <= -6;
  
  b[40] <= 4;
  b[41] <= 9;
  b[43] <= -5;
  b[44] <= -6;
  
  b[48] <= -4;
  b[50] <= -6;
  b[53] <= 10;
  b[54] <= -10;
  b[55] <= -8;
  
  b[56] <= 6;
  b[63] <= -8;
  
  #10;
  out_reg <= out;
  
  #10;
  for (i = 0; i < 64; i = i + 1) begin
    for (j = 0; j < 16; j = j + 1) begin
      buffer[`ML - 1 - j] = out_reg[((i + 1) * 16 - 1) - j];
    end
    if (buffer != exres4[i]) begin
      $display("Unexpected result in test 4: expected %d got %d!", exres4[i], buffer);
      $finish;
    end
  end
  $display("TEST 4 PASSED!");
  #10;
  
end
endmodule // top
