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
  arr[00]=-16'd166; arr[01]=-16'd7; arr[02]=-16'd4; arr[03]=-16'd4; arr[04]=16'd0; arr[05]=16'd0; arr[06]=16'd0; arr[07]=16'd0;\
  arr[08]=-16'd2; arr[09]=16'd0; arr[10]=16'd0; arr[11]=16'd0; arr[12]=16'd0; arr[13]=16'd0; arr[14]=16'd0; arr[15]=16'd0;\
  arr[16]=-16'd2; arr[17]=16'd0; arr[18]=16'd0; arr[19]=16'd0; arr[20]=16'd0; arr[21]=16'd0; arr[22]=16'd0; arr[23]=16'd0;\
  arr[24]=16'd0; arr[25]=16'd0; arr[26]=16'd0; arr[27]=16'd0; arr[28]=16'd0; arr[29]=16'd0; arr[30]=16'd0; arr[31]=16'd0;\
  arr[32]=16'd0; arr[33]=16'd0; arr[34]=16'd0; arr[35]=16'd0; arr[36]=16'd0; arr[37]=16'd0; arr[38]=16'd0; arr[39]=16'd0;\
  arr[40]=16'd0; arr[41]=16'd0; arr[42]=16'd0; arr[43]=16'd0; arr[44]=16'd0; arr[45]=16'd0; arr[46]=16'd0; arr[47]=16'd0;\
  arr[48]=16'd0; arr[49]=16'd0; arr[50]=16'd0; arr[51]=16'd0; arr[52]=16'd0; arr[53]=16'd0; arr[54]=16'd0; arr[55]=16'd0;\
  arr[56]=16'd0; arr[57]=16'd0; arr[58]=16'd0; arr[59]=16'd0; arr[60]=16'd0; arr[61]=16'd0; arr[62]=16'd0; arr[63]=16'd0

`define REF_INIT(arr)\
  arr[00]=-16'd24; arr[01]=-16'd23; arr[02]=-16'd21; arr[03]=-16'd21; arr[04]=-16'd21; arr[05]=-16'd21; arr[06]=-16'd21; arr[07]=-16'd20;\
  arr[08]=-16'd24; arr[09]=-16'd22; arr[10]=-16'd21; arr[11]=-16'd20; arr[12]=-16'd21; arr[13]=-16'd21; arr[14]=-16'd21; arr[15]=-16'd20;\
  arr[16]=-16'd23; arr[17]=-16'd22; arr[18]=-16'd21; arr[19]=-16'd20; arr[20]=-16'd20; arr[21]=-16'd21; arr[22]=-16'd20; arr[23]=-16'd20;\
  arr[24]=-16'd23; arr[25]=-16'd22; arr[26]=-16'd20; arr[27]=-16'd20; arr[28]=-16'd20; arr[29]=-16'd20; arr[30]=-16'd20; arr[31]=-16'd19;\
  arr[32]=-16'd23; arr[33]=-16'd22; arr[34]=-16'd20; arr[35]=-16'd20; arr[36]=-16'd20; arr[37]=-16'd20; arr[38]=-16'd20; arr[39]=-16'd19;\
  arr[40]=-16'd23; arr[41]=-16'd22; arr[42]=-16'd20; arr[43]=-16'd20; arr[44]=-16'd20; arr[45]=-16'd20; arr[46]=-16'd20; arr[47]=-16'd19;\
  arr[48]=-16'd23; arr[49]=-16'd22; arr[50]=-16'd20; arr[51]=-16'd20; arr[52]=-16'd20; arr[53]=-16'd20; arr[54]=-16'd20; arr[55]=-16'd19;\
  arr[56]=-16'd23; arr[57]=-16'd22; arr[58]=-16'd20; arr[59]=-16'd20; arr[60]=-16'd20; arr[61]=-16'd20; arr[62]=-16'd20; arr[63]=-16'd20

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
