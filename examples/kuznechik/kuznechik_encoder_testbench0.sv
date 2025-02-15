// Addition (integer): 2 stages each.
// XOR (integer): 2 stages each.
// Total: 128 stages.

`timescale 1s/1s

module MagmaEncoder_test0();

  localparam CIRCUIT_LATENCY = 128;

  reg [127:0] block;
  reg [255:0] key;
  reg [127:0] encoded;
  reg [127:0] expected;
  reg clk;

  MagmaEncoder inst (
    .block(block),
    .key(key),
    .encoded(encoded),
    .clk (clk)
  );

  initial clk = 0;

  always #1 clk = ~clk;

  initial begin

    @(negedge clk);
    $display("[MagmaEncoder: test 0] Input ready.");
    
    block = 128'1122334455667700ffeeddccbbaa9988;
    key = 256'8899aabbccddeeff0011223344556677fedcba98765432100123456789abcdef;
    expected = 128'h;
    $display("Input: [%0x], key: [%0x]", block, key);
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("MagmaEncoder_test0.vcd");
    $dumpvars(0, MagmaEncoder_test0);
    $display("[MagmaEncoder: test 0] Started...");

    $display("Output: %0h", encoded);
    if (expected == encoded) begin
      $display("GOOD: %0h == %0h", expected, encoded);
    end else begin
      $display("BAD: %0h != %0h", expected, encoded);
      $display("[MagmaEncoder: test 0] Stopped.");
      $finish;
    end

    $display("[MagmaEncoder: test 0] Stopped.");
    $finish;
  end

endmodule
