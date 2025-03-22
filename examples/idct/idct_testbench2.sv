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
  arr[00]=-`WIN'd166; arr[01]=-`WIN'd7; arr[02]=-`WIN'd4; arr[03]=-`WIN'd4; arr[04]=`WIN'd0; arr[05]=`WIN'd0; arr[06]=`WIN'd0; arr[07]=`WIN'd0;\
  arr[08]=-`WIN'd2; arr[09]=`WIN'd0; arr[10]=`WIN'd0; arr[11]=`WIN'd0; arr[12]=`WIN'd0; arr[13]=`WIN'd0; arr[14]=`WIN'd0; arr[15]=`WIN'd0;\
  arr[16]=-`WIN'd2; arr[17]=`WIN'd0; arr[18]=`WIN'd0; arr[19]=`WIN'd0; arr[20]=`WIN'd0; arr[21]=`WIN'd0; arr[22]=`WIN'd0; arr[23]=`WIN'd0;\
  arr[24]=`WIN'd0; arr[25]=`WIN'd0; arr[26]=`WIN'd0; arr[27]=`WIN'd0; arr[28]=`WIN'd0; arr[29]=`WIN'd0; arr[30]=`WIN'd0; arr[31]=`WIN'd0;\
  arr[32]=`WIN'd0; arr[33]=`WIN'd0; arr[34]=`WIN'd0; arr[35]=`WIN'd0; arr[36]=`WIN'd0; arr[37]=`WIN'd0; arr[38]=`WIN'd0; arr[39]=`WIN'd0;\
  arr[40]=`WIN'd0; arr[41]=`WIN'd0; arr[42]=`WIN'd0; arr[43]=`WIN'd0; arr[44]=`WIN'd0; arr[45]=`WIN'd0; arr[46]=`WIN'd0; arr[47]=`WIN'd0;\
  arr[48]=`WIN'd0; arr[49]=`WIN'd0; arr[50]=`WIN'd0; arr[51]=`WIN'd0; arr[52]=`WIN'd0; arr[53]=`WIN'd0; arr[54]=`WIN'd0; arr[55]=`WIN'd0;\
  arr[56]=`WIN'd0; arr[57]=`WIN'd0; arr[58]=`WIN'd0; arr[59]=`WIN'd0; arr[60]=`WIN'd0; arr[61]=`WIN'd0; arr[62]=`WIN'd0; arr[63]=`WIN'd0

`define REF_INIT(arr)\
  arr[00]=-`WOUT'd24; arr[01]=-`WOUT'd23; arr[02]=-`WOUT'd21; arr[03]=-`WOUT'd21; arr[04]=-`WOUT'd21; arr[05]=-`WOUT'd21; arr[06]=-`WOUT'd21; arr[07]=-`WOUT'd20;\
  arr[08]=-`WOUT'd24; arr[09]=-`WOUT'd22; arr[10]=-`WOUT'd21; arr[11]=-`WOUT'd20; arr[12]=-`WOUT'd21; arr[13]=-`WOUT'd21; arr[14]=-`WOUT'd21; arr[15]=-`WOUT'd20;\
  arr[16]=-`WOUT'd23; arr[17]=-`WOUT'd22; arr[18]=-`WOUT'd21; arr[19]=-`WOUT'd20; arr[20]=-`WOUT'd20; arr[21]=-`WOUT'd21; arr[22]=-`WOUT'd20; arr[23]=-`WOUT'd20;\
  arr[24]=-`WOUT'd23; arr[25]=-`WOUT'd22; arr[26]=-`WOUT'd20; arr[27]=-`WOUT'd20; arr[28]=-`WOUT'd20; arr[29]=-`WOUT'd20; arr[30]=-`WOUT'd20; arr[31]=-`WOUT'd19;\
  arr[32]=-`WOUT'd23; arr[33]=-`WOUT'd22; arr[34]=-`WOUT'd20; arr[35]=-`WOUT'd20; arr[36]=-`WOUT'd20; arr[37]=-`WOUT'd20; arr[38]=-`WOUT'd20; arr[39]=-`WOUT'd19;\
  arr[40]=-`WOUT'd23; arr[41]=-`WOUT'd22; arr[42]=-`WOUT'd20; arr[43]=-`WOUT'd20; arr[44]=-`WOUT'd20; arr[45]=-`WOUT'd20; arr[46]=-`WOUT'd20; arr[47]=-`WOUT'd19;\
  arr[48]=-`WOUT'd23; arr[49]=-`WOUT'd22; arr[50]=-`WOUT'd20; arr[51]=-`WOUT'd20; arr[52]=-`WOUT'd20; arr[53]=-`WOUT'd20; arr[54]=-`WOUT'd20; arr[55]=-`WOUT'd19;\
  arr[56]=-`WOUT'd23; arr[57]=-`WOUT'd22; arr[58]=-`WOUT'd20; arr[59]=-`WOUT'd20; arr[60]=-`WOUT'd20; arr[61]=-`WOUT'd20; arr[62]=-`WOUT'd20; arr[63]=-`WOUT'd20

`define ARRAY_TO_BITVECTOR(arr) \
  {arr[63], arr[62], arr[61], arr[60], arr[59], arr[58], arr[57], arr[56],\
  arr[55], arr[54], arr[53], arr[52], arr[51], arr[50], arr[49], arr[48],\
  arr[47], arr[46], arr[45], arr[44], arr[43], arr[42], arr[41], arr[40],\
  arr[39], arr[38], arr[37], arr[36], arr[35], arr[34], arr[33], arr[32],\
  arr[31], arr[30], arr[29], arr[28], arr[27], arr[26], arr[25], arr[24],\
  arr[23], arr[22], arr[21], arr[20], arr[19], arr[18], arr[17], arr[16],\
  arr[15], arr[14], arr[13], arr[12], arr[11], arr[10], arr[09], arr[08],\
  arr[07], arr[06], arr[05], arr[04], arr[03], arr[02], arr[01], arr[00]}

module IDCT_test2();

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
    $display("[IDCT: test 2] Input ready.");
    for (j = 0; j < 64; j = j + 1) begin
      x[j] = x_buffer[j];

      $display("Input[%0d]: [%0d]", j, x[j]);

    end
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("IDCT_test2.vcd");
    $dumpvars(0, IDCT_test2);
    $display("[IDCT: test 2] Started...");

    for (k = 0; k < 64; k = k + 1) begin
      $display("Output[%0d]: %0d", k, out[k]);
      if (expected[k] == out[k]) begin
        $display("GOOD: %0d == %0d", expected[k], out[k]);
      end else begin
        $display("BAD: %0d != %0d", expected[k], out[k]);
        $display("[IDCT: test 2] Stopped.");
        $finish;
      end
    end

    $display("[IDCT: test 2] Stopped.");
    $finish;
  end

endmodule
