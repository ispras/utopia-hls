//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

// Addition (integer): 1 stage each.
// Subtraction (integer): 1 stage each.
// Multiplication (integer): 3 stages each.
// Total: 26 stages.

`timescale 1s/1s

`define WIN 12
`define WOUT 9

`define INPUT_INIT(arr)\
  arr[00]=-`WIN'd240; arr[01]=`WIN'd8; arr[02]=-`WIN'd11; arr[03]=`WIN'd47; arr[04]=`WIN'd26; arr[05]=-`WIN'd6; arr[06]=`WIN'd0; arr[07]=`WIN'd5;\
  arr[08]=`WIN'd28; arr[09]=-`WIN'd6; arr[10]=`WIN'd85; arr[11]=`WIN'd44; arr[12]=-`WIN'd4; arr[13]=-`WIN'd25; arr[14]=`WIN'd5; arr[15]=`WIN'd16;\
  arr[16]=`WIN'd21; arr[17]=`WIN'd8; arr[18]=`WIN'd32; arr[19]=-`WIN'd16; arr[20]=-`WIN'd24; arr[21]=`WIN'd0; arr[22]=`WIN'd30; arr[23]=`WIN'd12;\
  arr[24]=-`WIN'd2; arr[25]=`WIN'd18; arr[26]=`WIN'd0; arr[27]=-`WIN'd2; arr[28]=`WIN'd0; arr[29]=`WIN'd7; arr[30]=`WIN'd0; arr[31]=-`WIN'd15;\
  arr[32]=`WIN'd7; arr[33]=`WIN'd4; arr[34]=`WIN'd15; arr[35]=-`WIN'd24; arr[36]=`WIN'd0; arr[37]=`WIN'd9; arr[38]=`WIN'd8; arr[39]=-`WIN'd6;\
  arr[40]=`WIN'd4; arr[41]=`WIN'd9; arr[42]=`WIN'd0; arr[43]=-`WIN'd5; arr[44]=-`WIN'd6; arr[45]=`WIN'd0; arr[46]=`WIN'd0; arr[47]=`WIN'd0;\
  arr[48]=-`WIN'd4; arr[49]=`WIN'd0; arr[50]=-`WIN'd6; arr[51]=`WIN'd0; arr[52]=`WIN'd0; arr[53]=`WIN'd10; arr[54]=-`WIN'd10; arr[55]=-`WIN'd8;\
  arr[56]=`WIN'd6; arr[57]=`WIN'd0; arr[58]=`WIN'd0; arr[59]=`WIN'd0; arr[60]=`WIN'd0; arr[61]=`WIN'd0; arr[62]=`WIN'd0; arr[63]=-`WIN'd8

`define REF_INIT(arr)\
  arr[00]=`WOUT'd21; arr[01]=-`WOUT'd10; arr[02]=-`WOUT'd26; arr[03]=-`WOUT'd61; arr[04]=-`WOUT'd43; arr[05]=-`WOUT'd17; arr[06]=-`WOUT'd22; arr[07]=-`WOUT'd8;\
  arr[08]=`WOUT'd5; arr[09]=-`WOUT'd28; arr[10]=-`WOUT'd47; arr[11]=-`WOUT'd73; arr[12]=-`WOUT'd11; arr[13]=-`WOUT'd14; arr[14]=-`WOUT'd24; arr[15]=-`WOUT'd17;\
  arr[16]=-`WOUT'd14; arr[17]=-`WOUT'd31; arr[18]=-`WOUT'd61; arr[19]=-`WOUT'd45; arr[20]=-`WOUT'd5; arr[21]=-`WOUT'd18; arr[22]=-`WOUT'd22; arr[23]=-`WOUT'd34;\
  arr[24]=-`WOUT'd23; arr[25]=-`WOUT'd36; arr[26]=-`WOUT'd49; arr[27]=-`WOUT'd32; arr[28]=-`WOUT'd12; arr[29]=-`WOUT'd33; arr[30]=-`WOUT'd33; arr[31]=-`WOUT'd35;\
  arr[32]=-`WOUT'd30; arr[33]=-`WOUT'd39; arr[34]=-`WOUT'd53; arr[35]=-`WOUT'd8; arr[36]=-`WOUT'd19; arr[37]=-`WOUT'd31; arr[38]=-`WOUT'd43; arr[39]=-`WOUT'd42;\
  arr[40]=-`WOUT'd41; arr[41]=-`WOUT'd43; arr[42]=-`WOUT'd50; arr[43]=-`WOUT'd4; arr[44]=-`WOUT'd15; arr[45]=-`WOUT'd33; arr[46]=-`WOUT'd44; arr[47]=-`WOUT'd66;\
  arr[48]=-`WOUT'd40; arr[49]=-`WOUT'd38; arr[50]=-`WOUT'd21; arr[51]=-`WOUT'd14; arr[52]=-`WOUT'd17; arr[53]=-`WOUT'd26; arr[54]=-`WOUT'd46; arr[55]=-`WOUT'd52;\
  arr[56]=-`WOUT'd44; arr[57]=-`WOUT'd47; arr[58]=-`WOUT'd9; arr[59]=-`WOUT'd12; arr[60]=-`WOUT'd30; arr[61]=-`WOUT'd33; arr[62]=-`WOUT'd38; arr[63]=-`WOUT'd37

`define ARRAY_TO_BITVECTOR(arr) \
  {arr[63], arr[62], arr[61], arr[60], arr[59], arr[58], arr[57], arr[56],\
  arr[55], arr[54], arr[53], arr[52], arr[51], arr[50], arr[49], arr[48],\
  arr[47], arr[46], arr[45], arr[44], arr[43], arr[42], arr[41], arr[40],\
  arr[39], arr[38], arr[37], arr[36], arr[35], arr[34], arr[33], arr[32],\
  arr[31], arr[30], arr[29], arr[28], arr[27], arr[26], arr[25], arr[24],\
  arr[23], arr[22], arr[21], arr[20], arr[19], arr[18], arr[17], arr[16],\
  arr[15], arr[14], arr[13], arr[12], arr[11], arr[10], arr[09], arr[08],\
  arr[07], arr[06], arr[05], arr[04], arr[03], arr[02], arr[01], arr[00]}

module IDCT_test3();

  localparam CIRCUIT_LATENCY = 26;

  reg signed [`WIN-1:0] x [63:0];
  reg signed [`WOUT-1:0] out [63:0];
  reg signed [`WIN-1:0] x_buffer [63:0];
  reg signed [`WOUT-1:0] expected [63:0];
  reg clk;

  IDCT inst (
    .x(`ARRAY_TO_BITVECTOR(x)),
    .out(`ARRAY_TO_BITVECTOR(out)),
    .clk (clk)
  );

  initial clk = 0;

  always #1 clk = ~clk;

  integer i;
  integer j;
  integer k;

  initial begin
    `INPUT_INIT(x_buffer);
    `REF_INIT(expected);
  end

  initial begin

    @(negedge clk);
    $display("[IDCT: test 3] Input ready.");
    for (j = 0; j < 64; j = j + 1) begin
      x[j] = x_buffer[j];

      $display("Input[%0d]: [%0d]", j, x[j]);

    end
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("IDCT_test3.vcd");
    $dumpvars(0, IDCT_test3);
    $display("[IDCT: test 3] Started...");

    for (k = 0; k < 64; k = k + 1) begin
      $display("Output[%0d]: %0d", k, out[k]);
      if (expected[k] == out[k]) begin
        $display("GOOD: %0d == %0d", expected[k], out[k]);
      end else begin
        $display("BAD: %0d != %0d", expected[k], out[k]);
        $display("[IDCT: test 3] Stopped.");
        $finish;
      end
    end

    $display("[IDCT: test 3] Stopped.");
    $finish;
  end

endmodule
