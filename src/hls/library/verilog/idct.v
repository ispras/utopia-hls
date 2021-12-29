
`define ML 16
`define W1 2841
`define W2 2676
`define W3 2408
`define W5 1609
`define W6 1108
`define W7 565

`define G0 'h000000000000000100000001ffff00060000ffff0000ffff0001fffd0004fff50001000200000003fffe0006fff9001afffffffe0001fffc0003fff8000bffdf00020004ffff0007fffb000dffef003cfffefffb0001fff70007fff00017ffb90005000cfffd0015fff10027ffcc00b0fffcfff40005ffea0013ffd6003fff53

`define G1 'h000000000000ffff0000ffff0001fffa0000000100000001ffff0003fffc000bfffffffe0000fffd0002fffa0007ffe600010002ffff0004fffd0008fff50021fffefffc0001fff90005fff30011ffc400020005ffff0009fff90010ffe90047fffbfff40003ffeb000fffd90034ff500004000cfffb0016ffed002affc100ad

`define G2 'h0003000300030003000300030003000200030003000300030003000300030002000300030003000300030003000300020003000300030003000300030003000200030003000300030003000300030002000300030003000300030003000300020003000300030003000300030003000200030003000300030003000300030002

`define G3 'h0003000300030002000100010000000000030003000300020001000100000000000300030002000200010001000000000003000300020002000100010001000000030003000200020001000100010001000300020002000200010001000100010002000200020002000100010001000100020002000200020001000100010001

`define G4 'hffecffecffecffecffecffecffeaffe9ffedffecffecffecffecffecffeaffe9ffedffecffecffecffecffecffeaffe9ffedffecffecffecffecffecffeaffe9ffedffecffecffecffecffecffeaffe9ffecffecffebffecffecffebffeaffe9ffecffebffebffebffecffebffeaffe8ffecffebffebffebffebffebffe9ffe8

`define G5 'hffdbffdaffdfffe2fff4fff7ffd1ffd4ffccffd2ffe6ffeffff2ffebffdaffd8ffbeffd4ffdffff1fffcffceffd5ffd7ffd6ffd5ffe1ffedfff8ffcbffd9ffe2ffddffdfffdffff4ffe0ffcfffdcffe9ffdeffeaffeefffbffd3ffc3ffe1fff2ffefffe8fff2fff5ffb7ffd1ffe40005fff8ffeaffefffd5ffc3ffe6fff60015

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
reg signed [`ML-1:0] b [63:0];
wire [`ML*8*8-1:0] out;
reg [`ML*8*8-1:0] out_reg;
Fast_IDCT idct({b[63], b[62], b[61], b[60], b[59], b[58], b[57], b[56],
                b[55], b[54], b[53], b[52], b[51], b[50], b[49], b[48],
                b[47], b[46], b[45], b[44], b[43], b[42], b[41], b[40],
                b[39], b[38], b[37], b[36], b[35], b[34], b[33], b[32],
                b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                b[23], b[22], b[21], b[20], b[19], b[18], b[17], b[16],
                b[15], b[14], b[13], b[12], b[11], b[10], b[09], b[08],
                b[07], b[06], b[05], b[04], b[03], b[02], b[01], b[00]}, out);
initial begin
  $dumpfile("test.vcd");
  $dumpvars(6, top);

  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= -1*i;
  end

  #10;
  out_reg <= out;

  #10;
  if (out_reg != `G0) begin
    $display("Error in assert №0!");
    $finish;
  end
  #10;
  $display("out_reg (hex) = %x", out_reg);

  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 1*i;
  end

  #10;
  out_reg <= out;
  
  #10;
  if (out_reg != `G1) begin
    $display("Error in assert №1!");
    $finish;
  end
  #10;
  $display("out_reg (hex) = %x", out_reg);
  
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  
  b[0] <= 23;
  b[1] <= -1;
  b[2] <= -2;
  
  #10;
  out_reg <= out;
  
  #10;
  if (out_reg != `G2) begin
    $display("Error in assert №2!");
    $finish;
  end
  #10;
  $display("out_reg (hex) = %x", out_reg);
  
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  b[0] <= 13;
  b[1] <= -7;
  b[9] <= 2;
  
  #10;
  out_reg <= out;
  
  #10;
  if (out_reg != `G3) begin
    $display("Error in assert №3!");
    $finish;
  end
  #10;
  $display("out_reg (hex) = %x", out_reg);

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
  if (out_reg != `G4) begin
    $display("Error in assert №4!");
    $finish;
  end
  #10;
  $display("out_reg (hex) = %x", out_reg);
  
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
  if (out_reg != `G5) begin
    $display("Error in assert №5!");
    $finish;
  end
  #10;
  $display("out_reg (hex) = %x", out_reg);
  

end
endmodule // top
