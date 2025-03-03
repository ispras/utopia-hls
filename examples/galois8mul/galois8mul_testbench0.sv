//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

// XOR (integer): 1 stage each.
// Total: 9 stages.

`timescale 1s/1s

module Galois8Mul_test0();

  localparam CIRCUIT_LATENCY = 9;

  reg [7:0] left;
  reg [7:0] right;
  reg [7:0] result;
  reg [7:0] expected;
  reg clk;

  Galois8Mul inst (
    .left(left),
    .right(right),
    .result(result),
    .clk(clk)
  );

  initial clk = 0;

  always #1 clk = ~clk;

  initial begin

    @(negedge clk);
    $display("[Galois8Mul: test 0] Input ready.");
    
    left = 8'h72;
    right = 8'h69;
    expected = 8'h8c;
    $display("Input: [%0h, %0h]", left, right);
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("Galois8Mul_test0.vcd");
    $dumpvars(0, Galois8Mul_test0);
    $display("[Galois8Mul: test 0] Started...");

    $display("Output: %0h", result);
    if (expected == result) begin
      $display("GOOD: %0h == %0h", expected, result);
    end else begin
      $display("BAD: %0h != %0h", expected, result);
      $display("[Galois8Mul: test 0] Stopped.");
      $finish;
    end

    $display("[Galois8Mul: test 0] Stopped.");
    $finish;
  end

endmodule
