/*
 * Copyright 2022 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 */

 /**********************************************************/
 /* inverse two dimensional DCT, Chen-Wang algorithm       */
 /* (cf. IEEE ASSP-32, pp. 803-816, Aug. 1984)             */
 /* 32-bit integer arithmetic (8 bit coefficients)         */
 /* 11 mults, 29 adds per DCT                              */
 /*                                      sE, 18.8.91       */
 /**********************************************************/
 /* coefficients extended to 12 bit for IEEE1180-1990      */
 /* compliance                           sE,  2.1.94       */
 /**********************************************************/

// Input width
`define WIN 12
// Width of communication channel between calculators for rows and columns
`define WIM 13
// Output width
`define WOUT 9
// IDCT coefficients
`define W1 2841
`define W2 2676
`define W3 2408
`define W5 1609
`define W6 1108
`define W7 565

// Reference values for tests
`define REF0 {9'h0,   9'h0,   9'h0,   9'h1,   9'h0,   9'h1,   9'h1ff, 9'h6,\
              9'h0,   9'h1ff, 9'h0,   9'h1ff, 9'h1,   9'h1fd, 9'h4,   9'h1f5,\
              9'h1,   9'h2,   9'h0,   9'h3,   9'h1fe, 9'h6,   9'h1f9, 9'h1a,\
              9'h1ff, 9'h1fe, 9'h1,   9'h1fc, 9'h3,   9'h1f8, 9'hb,   9'h1df,\
              9'h2,   9'h4,   9'h1ff, 9'h7,   9'h1fb, 9'hd,   9'h1ef, 9'h3c,\
              9'h1fe, 9'h1fb, 9'h1,   9'h1f7, 9'h7,   9'h1f0, 9'h17,  9'h1b9,\
              9'h5,   9'hc,   9'h1fd, 9'h15,  9'h1f1, 9'h27,  9'h1cc, 9'hb0,\
              9'h1fc, 9'h1f4, 9'h5,   9'h1ea, 9'h13,  9'h1d6, 9'h3f,  9'h153}

`define REF1 {9'h0,   9'h0,   9'h0,   9'h1ff, 9'h0,   9'h1ff, 9'h1,   9'h1fa,\
              9'h0,   9'h1,   9'h0,   9'h1,   9'h1ff, 9'h3,   9'h1fc, 9'hb,\
              9'h1ff, 9'h1fe, 9'h0,   9'h1fd, 9'h2,   9'h1fa, 9'h7,   9'h1e6,\
              9'h1,   9'h2,   9'h1ff, 9'h4,   9'h1fd, 9'h8,   9'h1f5, 9'h21,\
              9'h1fe, 9'h1fc, 9'h1,   9'h1f9, 9'h5,   9'h1f3, 9'h11,  9'h1c4,\
              9'h2,   9'h5,   9'h1ff, 9'h9,   9'h1f9, 9'h10,  9'h1e9, 9'h47,\
              9'h1fb, 9'h1f4, 9'h3,   9'h1eb, 9'hf,   9'h1d9, 9'h34,  9'h150,\
              9'h4,   9'hc,   9'h1fb, 9'h16,  9'h1ed, 9'h2a,  9'h1c1, 9'had}

`define REF2 {9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2,\
              9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h3, 9'h2}

`define REF3 {9'h3, 9'h3, 9'h3, 9'h2, 9'h1, 9'h1, 9'h0, 9'h0,\
              9'h3, 9'h3, 9'h3, 9'h2, 9'h1, 9'h1, 9'h0, 9'h0,\
              9'h3, 9'h3, 9'h2, 9'h2, 9'h1, 9'h1, 9'h0, 9'h0,\
              9'h3, 9'h3, 9'h2, 9'h2, 9'h1, 9'h1, 9'h1, 9'h0,\
              9'h3, 9'h3, 9'h2, 9'h2, 9'h1, 9'h1, 9'h1, 9'h1,\
              9'h3, 9'h2, 9'h2, 9'h2, 9'h1, 9'h1, 9'h1, 9'h1,\
              9'h2, 9'h2, 9'h2, 9'h2, 9'h1, 9'h1, 9'h1, 9'h1,\
              9'h2, 9'h2, 9'h2, 9'h2, 9'h1, 9'h1, 9'h1, 9'h1}

`define REF4 {9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ea, 9'h1e9,\
              9'h1ed, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ea, 9'h1e9,\
              9'h1ed, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ea, 9'h1e9,\
              9'h1ed, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ea, 9'h1e9,\
              9'h1ed, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ec, 9'h1ea, 9'h1e9,\
              9'h1ec, 9'h1ec, 9'h1eb, 9'h1ec, 9'h1ec, 9'h1eb, 9'h1ea, 9'h1e9,\
              9'h1ec, 9'h1eb, 9'h1eb, 9'h1eb, 9'h1ec, 9'h1eb, 9'h1ea, 9'h1e8,\
              9'h1ec, 9'h1eb, 9'h1eb, 9'h1eb, 9'h1eb, 9'h1eb, 9'h1e9, 9'h1e8}

`define IN5 { -12'd8,   12'd0,   12'd0,   12'd0,   12'd0,   12'd0,   12'd0,   12'd6,\
              -12'd8,  -12'd10,  12'd10,  12'd0,   12'd0,  -12'd6,   12'd0,  -12'd4,\
               12'd0,   12'd0,   12'd0,  -12'd6,  -12'd5,   12'd0,   12'd9,   12'd4,\
              -12'd6,   12'd8,   12'd9,   12'd0,  -12'd24,  12'd15,  12'd4,   12'd7,\
              -12'd15,  12'd0,   12'd7,   12'd0,  -12'd2,   12'd0,   12'd18, -12'd2,\
               12'd12,  12'd30,  12'd0,  -12'd24, -12'd16,  12'd32,  12'd8,   12'd21,\
               12'd16,  12'd5,  -12'd25, -12'd4,   12'd44,  12'd85, -12'd6,   12'd28,\
               12'd5,   12'd0,  -12'd6,   12'd26,  12'd47, -12'd11,  12'd8,  -12'd240}

`define REF5 {9'h1db, 9'h1da, 9'h1df, 9'h1e2, 9'h1f4, 9'h1f7, 9'h1d1, 9'h1d4,\
              9'h1cc, 9'h1d2, 9'h1e6, 9'h1ef, 9'h1f2, 9'h1eb, 9'h1da, 9'h1d8,\
              9'h1be, 9'h1d4, 9'h1df, 9'h1f1, 9'h1fc, 9'h1ce, 9'h1d5, 9'h1d7,\
              9'h1d6, 9'h1d5, 9'h1e1, 9'h1ed, 9'h1f8, 9'h1cb, 9'h1d9, 9'h1e2,\
              9'h1dd, 9'h1df, 9'h1df, 9'h1f4, 9'h1e0, 9'h1cf, 9'h1dc, 9'h1e9,\
              9'h1de, 9'h1ea, 9'h1ee, 9'h1fb, 9'h1d3, 9'h1c3, 9'h1e1, 9'h1f2,\
              9'h1ef, 9'h1e8, 9'h1f2, 9'h1f5, 9'h1b7, 9'h1d1, 9'h1e4, 9'h005,\
              9'h1f8, 9'h1ea, 9'h1ef, 9'h1d5, 9'h1c3, 9'h1e6, 9'h1f6, 9'h015}

// Map of 64-element array to the correspondent bit vector
`define ARRAY_TO_BITVECTOR(a) \
 {a[63], a[62], a[61], a[60], a[59], a[58], a[57], a[56],\
  a[55], a[54], a[53], a[52], a[51], a[50], a[49], a[48],\
  a[47], a[46], a[45], a[44], a[43], a[42], a[41], a[40],\
  a[39], a[38], a[37], a[36], a[35], a[34], a[33], a[32],\
  a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],\
  a[23], a[22], a[21], a[20], a[19], a[18], a[17], a[16],\
  a[15], a[14], a[13], a[12], a[11], a[10], a[09], a[08],\
  a[07], a[06], a[05], a[04], a[03], a[02], a[01], a[00]}

// Test check
`define TEST(number) \
  #1; \
  out_reg <= out; \
  #1; \
  if (out_reg != `REF``number) begin \
    $display("[FAIL] test #number:"); \
    $display("expected value is 0x%x\nreceived value is 0x%x", `REF``number, out_reg); \
    $finish; \
  end else begin \
    $display("[OK] test #number: out_reg is 0x%x", out_reg); \
  end

module idctrow(input [`WIN-1:0] i0, input [`WIN-1:0] i1, input [`WIN-1:0] i2, input [`WIN-1:0] i3,
               input [`WIN-1:0] i4, input [`WIN-1:0] i5, input [`WIN-1:0] i6, input [`WIN-1:0] i7,
               output [`WIM-1:0] b0, output [`WIM-1:0] b1, output [`WIM-1:0] b2, output [`WIM-1:0] b3,
               output [`WIM-1:0] b4, output [`WIM-1:0] b5, output [`WIM-1:0] b6, output [`WIM-1:0] b7);

wire signed [`WIN*2-1:0] x [7:0][3:0];
wire return_zero;

wire [`WIM-1:0] b0;
wire [`WIM-1:0] b1;
wire [`WIM-1:0] b2;
wire [`WIM-1:0] b3;
wire [`WIM-1:0] b4;
wire [`WIM-1:0] b5;
wire [`WIM-1:0] b6;
wire [`WIM-1:0] b7;

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
wire signed [`WIN*2-1:0] tmp0;
wire signed [`WIN*2-1:0] tmp1;
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
wire signed [`WIN*2-1:0] tmp2;
wire signed [`WIN*2-1:0] tmp3;
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
wire signed [`WIN*2-1:0] tmp4;
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
wire signed [`WIN*2-1:0] tmp5;
assign tmp5 = $signed(i0) << 3;
assign b0 = br ? tmp5[`WIM-1:0] : (x[7][3] + x[1][3]) >>> 8;
assign b1 = br ? tmp5[`WIM-1:0] : (x[3][3] + x[2][3]) >>> 8;
assign b2 = br ? tmp5[`WIM-1:0] : (x[0][3] + x[4][3]) >>> 8;
assign b3 = br ? tmp5[`WIM-1:0] : (   tmp4 + x[6][3]) >>> 8;
assign b4 = br ? tmp5[`WIM-1:0] : (   tmp4 - x[6][3]) >>> 8;
assign b5 = br ? tmp5[`WIM-1:0] : (x[0][3] - x[4][3]) >>> 8;
assign b6 = br ? tmp5[`WIM-1:0] : (x[3][3] - x[2][3]) >>> 8;
assign b7 = br ? tmp5[`WIM-1:0] : (x[7][3] - x[1][3]) >>> 8;

endmodule // idctrow

module idctcol(input [`WIM-1:0] i0, input [`WIM-1:0] i1, input [`WIM-1:0] i2, input [`WIM-1:0] i3,
               input [`WIM-1:0] i4, input [`WIM-1:0] i5, input [`WIM-1:0] i6, input [`WIM-1:0] i7,
               output [`WOUT-1:0] b0, output [`WOUT-1:0] b1, output [`WOUT-1:0] b2, output [`WOUT-1:0] b3,
               output [`WOUT-1:0] b4, output [`WOUT-1:0] b5, output [`WOUT-1:0] b6, output [`WOUT-1:0] b7);

wire signed [`WIM*2-1:0] x [7:0][3:0];
wire return_zero;

wire [`WOUT-1:0] b0;
wire [`WOUT-1:0] b1;
wire [`WOUT-1:0] b2;
wire [`WOUT-1:0] b3;
wire [`WOUT-1:0] b4;
wire [`WOUT-1:0] b5;
wire [`WOUT-1:0] b6;
wire [`WOUT-1:0] b7;

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
wire signed [`WIM*2-1:0] tmp0;
wire signed [`WIM*2-1:0] tmp1;
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
wire signed [`WIM*2-1:0] tmp2;
wire signed [`WIM*2-1:0] tmp3;
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
wire signed [`WIM*2-1:0] tmp4;
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
wire signed [`WOUT-1:0] tmp5 = iclp13(($signed(i0) + 32) >>> 6);
assign b0 = br ? tmp5 : iclp13((x[7][3] + x[1][3]) >>> 14);
assign b1 = br ? tmp5 : iclp13((x[3][3] + x[2][3]) >>> 14);
assign b2 = br ? tmp5 : iclp13((x[0][3] + x[4][3]) >>> 14);
assign b3 = br ? tmp5 : iclp13((   tmp4 + x[6][3]) >>> 14);
assign b4 = br ? tmp5 : iclp13((   tmp4 - x[6][3]) >>> 14);
assign b5 = br ? tmp5 : iclp13((x[0][3] - x[4][3]) >>> 14);
assign b6 = br ? tmp5 : iclp13((x[3][3] - x[2][3]) >>> 14);
assign b7 = br ? tmp5 : iclp13((x[7][3] - x[1][3]) >>> 14);

function [`WOUT-1:0] iclp13;
  input [12:0] in;
  begin
    if (in[12] == 1 && in[11:8] != 4'hF)
      iclp13 = `WOUT'h80;
    else if (in[12] == 0 && in[11:8] != 0)
      iclp13 = `WOUT'h7F;
    else
      iclp13 = in[`WOUT-1:0];
  end
endfunction

endmodule // idctcol


module Fast_IDCT(input [`WIN*8*8-1:0] in, output [`WOUT*8*8-1:0] out);

wire [`WIM*8*8-1:0] rtc;
wire [`WOUT*8*8-1:0] out;

genvar i;
generate for (i=1; i<8*8; i=i+8) begin
    idctrow ir(in[(i+0)*`WIN-1 : (i-1)*`WIN],
               in[(i+1)*`WIN-1 : (i+0)*`WIN],
               in[(i+2)*`WIN-1 : (i+1)*`WIN],
               in[(i+3)*`WIN-1 : (i+2)*`WIN],
               in[(i+4)*`WIN-1 : (i+3)*`WIN],
               in[(i+5)*`WIN-1 : (i+4)*`WIN],
               in[(i+6)*`WIN-1 : (i+5)*`WIN],
               in[(i+7)*`WIN-1 : (i+6)*`WIN],
               rtc[(i+0)*`WIM-1 : (i-1)*`WIM],
               rtc[(i+1)*`WIM-1 : (i+0)*`WIM],
               rtc[(i+2)*`WIM-1 : (i+1)*`WIM],
               rtc[(i+3)*`WIM-1 : (i+2)*`WIM],
               rtc[(i+4)*`WIM-1 : (i+3)*`WIM],
               rtc[(i+5)*`WIM-1 : (i+4)*`WIM],
               rtc[(i+6)*`WIM-1 : (i+5)*`WIM],
               rtc[(i+7)*`WIM-1 : (i+6)*`WIM]);
end
endgenerate
generate for (i=1; i<=8; i=i+1) begin
    idctcol ic(rtc[(i+8*0)*`WIM-1 : (i-1+8*0)*`WIM],
               rtc[(i+8*1)*`WIM-1 : (i-1+8*1)*`WIM],
               rtc[(i+8*2)*`WIM-1 : (i-1+8*2)*`WIM],
               rtc[(i+8*3)*`WIM-1 : (i-1+8*3)*`WIM],
               rtc[(i+8*4)*`WIM-1 : (i-1+8*4)*`WIM],
               rtc[(i+8*5)*`WIM-1 : (i-1+8*5)*`WIM],
               rtc[(i+8*6)*`WIM-1 : (i-1+8*6)*`WIM],
               rtc[(i+8*7)*`WIM-1 : (i-1+8*7)*`WIM],
               out[(i+8*0)*`WOUT-1 : (i-1+8*0)*`WOUT],
               out[(i+8*1)*`WOUT-1 : (i-1+8*1)*`WOUT],
               out[(i+8*2)*`WOUT-1 : (i-1+8*2)*`WOUT],
               out[(i+8*3)*`WOUT-1 : (i-1+8*3)*`WOUT],
               out[(i+8*4)*`WOUT-1 : (i-1+8*4)*`WOUT],
               out[(i+8*5)*`WOUT-1 : (i-1+8*5)*`WOUT],
               out[(i+8*6)*`WOUT-1 : (i-1+8*6)*`WOUT],
               out[(i+8*7)*`WOUT-1 : (i-1+8*7)*`WOUT]);
end
endgenerate
endmodule // Fast_IDCT

module top ();
integer i;
reg signed [`WIN-1:0] b [63:0];
wire [`WOUT*8*8-1:0] out;
reg [`WOUT*8*8-1:0] out_reg;

Fast_IDCT idct(`ARRAY_TO_BITVECTOR(b), out);

initial begin
  $dumpfile("test.vcd");
  $dumpvars(6, top);

  // TEST 0
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= -1*i;
  end
  `TEST(0);

  // TEST 1
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 1*i;
  end
  `TEST(1);

  // TEST 2
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  b[0] <= 23;
  b[1] <= -1;
  b[2] <= -2;
  `TEST(2);

  // TEST 3
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  b[0] <= 13;
  b[1] <= -7;
  b[9] <= 2;
  `TEST(3);
  
  // TEST 4
  for (i = 0; i < 64; i = i + 1) begin
    b[i] <= 0;
  end
  b[0] <= -166;
  b[1] <= -7;
  b[2] <= -4;
  b[3] <= -4;
  b[8] <= -2;
  b[16] <= -2;
  `TEST(4);

  // TEST 5
  `ARRAY_TO_BITVECTOR(b) <= `IN5;
  `TEST(5);

  $display("[SUCCESS] Tests passed!");
end
endmodule // top
