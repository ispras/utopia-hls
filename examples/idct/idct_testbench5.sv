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
  arr[00]=`WIN'd0; arr[01]=-`WIN'd1; arr[02]=-`WIN'd2; arr[03]=-`WIN'd3; arr[04]=-`WIN'd4; arr[05]=-`WIN'd5; arr[06]=-`WIN'd6; arr[07]=-`WIN'd7;\
  arr[08]=-`WIN'd8; arr[09]=-`WIN'd9; arr[10]=-`WIN'd10; arr[11]=-`WIN'd11; arr[12]=-`WIN'd12; arr[13]=-`WIN'd13; arr[14]=-`WIN'd14; arr[15]=-`WIN'd15;\
  arr[16]=-`WIN'd16; arr[17]=-`WIN'd17; arr[18]=-`WIN'd18; arr[19]=-`WIN'd19; arr[20]=-`WIN'd20; arr[21]=-`WIN'd21; arr[22]=-`WIN'd22; arr[23]=-`WIN'd23;\
  arr[24]=-`WIN'd24; arr[25]=-`WIN'd25; arr[26]=-`WIN'd26; arr[27]=-`WIN'd27; arr[28]=-`WIN'd28; arr[29]=-`WIN'd29; arr[30]=-`WIN'd30; arr[31]=-`WIN'd31;\
  arr[32]=-`WIN'd32; arr[33]=-`WIN'd33; arr[34]=-`WIN'd34; arr[35]=-`WIN'd35; arr[36]=-`WIN'd36; arr[37]=-`WIN'd37; arr[38]=-`WIN'd38; arr[39]=-`WIN'd39;\
  arr[40]=-`WIN'd40; arr[41]=-`WIN'd41; arr[42]=-`WIN'd42; arr[43]=-`WIN'd43; arr[44]=-`WIN'd44; arr[45]=-`WIN'd45; arr[46]=-`WIN'd46; arr[47]=-`WIN'd47;\
  arr[48]=-`WIN'd48; arr[49]=-`WIN'd49; arr[50]=-`WIN'd50; arr[51]=-`WIN'd51; arr[52]=-`WIN'd52; arr[53]=-`WIN'd53; arr[54]=-`WIN'd54; arr[55]=-`WIN'd55;\
  arr[56]=-`WIN'd56; arr[57]=-`WIN'd57; arr[58]=-`WIN'd58; arr[59]=-`WIN'd59; arr[60]=-`WIN'd60; arr[61]=-`WIN'd61; arr[62]=-`WIN'd62; arr[63]=-`WIN'd63

`define REF_INIT(arr)\
  arr[00]=-`WOUT'd173; arr[01]=`WOUT'd63; arr[02]=-`WOUT'd42; arr[03]=`WOUT'd19; arr[04]=-`WOUT'd22; arr[05]=`WOUT'd5; arr[06]=-`WOUT'd12; arr[07]=-`WOUT'd4;\
  arr[08]=`WOUT'd176; arr[09]=-`WOUT'd52; arr[10]=`WOUT'd39; arr[11]=-`WOUT'd15; arr[12]=`WOUT'd21; arr[13]=-`WOUT'd3; arr[14]=`WOUT'd12; arr[15]=`WOUT'd5;\
  arr[16]=-`WOUT'd71; arr[17]=`WOUT'd23; arr[18]=-`WOUT'd16; arr[19]=`WOUT'd7; arr[20]=-`WOUT'd9; arr[21]=`WOUT'd1; arr[22]=-`WOUT'd5; arr[23]=-`WOUT'd2;\
  arr[24]=`WOUT'd60; arr[25]=-`WOUT'd17; arr[26]=`WOUT'd13; arr[27]=-`WOUT'd5; arr[28]=`WOUT'd7; arr[29]=-`WOUT'd1; arr[30]=`WOUT'd4; arr[31]=`WOUT'd2;\
  arr[32]=-`WOUT'd33; arr[33]=`WOUT'd11; arr[34]=-`WOUT'd8; arr[35]=`WOUT'd3; arr[36]=-`WOUT'd4; arr[37]=`WOUT'd1; arr[38]=-`WOUT'd2; arr[39]=-`WOUT'd1;\
  arr[40]=`WOUT'd26; arr[41]=-`WOUT'd7; arr[42]=`WOUT'd6; arr[43]=-`WOUT'd2; arr[44]=`WOUT'd3; arr[45]=`WOUT'd0; arr[46]=`WOUT'd2; arr[47]=`WOUT'd1;\
  arr[48]=-`WOUT'd11; arr[49]=`WOUT'd4; arr[50]=-`WOUT'd3; arr[51]=`WOUT'd1; arr[52]=-`WOUT'd1; arr[53]=`WOUT'd0; arr[54]=-`WOUT'd1; arr[55]=`WOUT'd0;\
  arr[56]=`WOUT'd6; arr[57]=-`WOUT'd1; arr[58]=`WOUT'd1; arr[59]=`WOUT'd0; arr[60]=`WOUT'd1; arr[61]=`WOUT'd0; arr[62]=`WOUT'd0; arr[63]=`WOUT'd0

`define ARRAY_TO_BITVECTOR(arr) \
  {arr[63], arr[62], arr[61], arr[60], arr[59], arr[58], arr[57], arr[56],\
  arr[55], arr[54], arr[53], arr[52], arr[51], arr[50], arr[49], arr[48],\
  arr[47], arr[46], arr[45], arr[44], arr[43], arr[42], arr[41], arr[40],\
  arr[39], arr[38], arr[37], arr[36], arr[35], arr[34], arr[33], arr[32],\
  arr[31], arr[30], arr[29], arr[28], arr[27], arr[26], arr[25], arr[24],\
  arr[23], arr[22], arr[21], arr[20], arr[19], arr[18], arr[17], arr[16],\
  arr[15], arr[14], arr[13], arr[12], arr[11], arr[10], arr[09], arr[08],\
  arr[07], arr[06], arr[05], arr[04], arr[03], arr[02], arr[01], arr[00]}

module IDCT_test5();

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
    $display("[IDCT: test 5] Input ready.");
    for (j = 0; j < 64; j = j + 1) begin
      x[j] = x_buffer[j];

      $display("Input[%0d]: [%0d]", j, x[j]);

    end
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("IDCT_test5.vcd");
    $dumpvars(0, IDCT_test5);
    $display("[IDCT: test 5] Started...");

    for (k = 0; k < 64; k = k + 1) begin
      $display("Output[%0d]: %0d", k, out[k]);
      if (expected[k] == out[k]) begin
        $display("GOOD: %0d == %0d", expected[k], out[k]);
      end else begin
        $display("BAD: %0d != %0d", expected[k], out[k]);
        $display("[IDCT: test 5] Stopped.");
        $finish;
      end
    end

    $display("[IDCT: test 5] Stopped.");
    $finish;
  end

endmodule
