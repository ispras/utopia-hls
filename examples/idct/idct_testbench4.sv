// Addition (integer): 1 stage each.
// Subtraction (integer): 1 stage each.
// Multiplication (integer): 3 stages each.
// "Greater" comparison (integer): 3 stages each.
// "Greater or equal" comparison (integer): 3 stages each.
// Total: 29 stages.

`timescale 1s/1s

`define INPUT_INIT(arr)\
  arr[00]=16'd0; arr[01]=16'd1; arr[02]=16'd2; arr[03]=16'd3; arr[04]=16'd4; arr[05]=16'd5; arr[06]=16'd6; arr[07]=16'd7;\
  arr[08]=16'd8; arr[09]=16'd9; arr[10]=16'd10; arr[11]=16'd11; arr[12]=16'd12; arr[13]=16'd13; arr[14]=16'd14; arr[15]=16'd15;\
  arr[16]=16'd16; arr[17]=16'd17; arr[18]=16'd18; arr[19]=16'd19; arr[20]=16'd20; arr[21]=16'd21; arr[22]=16'd22; arr[23]=16'd23;\
  arr[24]=16'd24; arr[25]=16'd25; arr[26]=16'd26; arr[27]=16'd27; arr[28]=16'd28; arr[29]=16'd29; arr[30]=16'd30; arr[31]=16'd31;\
  arr[32]=16'd32; arr[33]=16'd33; arr[34]=16'd34; arr[35]=16'd35; arr[36]=16'd36; arr[37]=16'd37; arr[38]=16'd38; arr[39]=16'd39;\
  arr[40]=16'd40; arr[41]=16'd41; arr[42]=16'd42; arr[43]=16'd43; arr[44]=16'd44; arr[45]=16'd45; arr[46]=16'd46; arr[47]=16'd47;\
  arr[48]=16'd48; arr[49]=16'd49; arr[50]=16'd50; arr[51]=16'd51; arr[52]=16'd52; arr[53]=16'd53; arr[54]=16'd54; arr[55]=16'd55;\
  arr[56]=16'd56; arr[57]=16'd57; arr[58]=16'd58; arr[59]=16'd59; arr[60]=16'd60; arr[61]=16'd61; arr[62]=16'd62; arr[63]=16'd63

`define REF_INIT(arr)\
  arr[00]=16'had; arr[01]=16'hffc1; arr[02]=16'h2a; arr[03]=16'hffed; arr[04]=16'h16; arr[05]=16'hfffb; arr[06]=16'hc; arr[07]=16'h4;\
  arr[08]=16'hff50; arr[09]=16'h34; arr[10]=16'hffd9; arr[11]=16'hf; arr[12]=16'hffeb; arr[13]=16'h3; arr[14]=16'hfff4; arr[15]=16'hfffb;\
  arr[16]=16'h47; arr[17]=16'hffe9; arr[18]=16'h10; arr[19]=16'hfff9; arr[20]=16'h9; arr[21]=16'hffff; arr[22]=16'h5; arr[23]=16'h2;\
  arr[24]=16'hffc4; arr[25]=16'h11; arr[26]=16'hfff3; arr[27]=16'h5; arr[28]=16'hfff9; arr[29]=16'h1; arr[30]=16'hfffc; arr[31]=16'hfffe;\
  arr[32]=16'h21; arr[33]=16'hfff5; arr[34]=16'h8; arr[35]=16'hfffd; arr[36]=16'h4; arr[37]=16'hffff; arr[38]=16'h2; arr[39]=16'h1;\
  arr[40]=16'hffe6; arr[41]=16'h7; arr[42]=16'hfffa; arr[43]=16'h2; arr[44]=16'hfffd; arr[45]=16'h0; arr[46]=16'hfffe; arr[47]=16'hffff;\
  arr[48]=16'hb; arr[49]=16'hfffc; arr[50]=16'h3; arr[51]=16'hffff; arr[52]=16'h1; arr[53]=16'h0; arr[54]=16'h1; arr[55]=16'h0;\
  arr[56]=16'hfffa; arr[57]=16'h1; arr[58]=16'hffff; arr[59]=16'h0; arr[60]=16'hffff; arr[61]=16'h0; arr[62]=16'h0; arr[63]=16'h0

`define ARRAY_BINDING(port, arr)\
  .``port``0(``arr``[00]), .``port``1(``arr``[01]), .``port``2(``arr``[02]), .``port``3(``arr``[03]), .``port``4(``arr``[04]), .``port``5(``arr``[05]), .``port``6(``arr``[06]), .``port``7(``arr``[07]),\
  .``port``8(``arr``[08]), .``port``9(``arr``[09]), .``port``10(``arr``[10]), .``port``11(``arr``[11]), .``port``12(``arr``[12]), .``port``13(``arr``[13]), .``port``14(``arr``[14]), .``port``15(``arr``[15]),\
  .``port``16(``arr``[16]), .``port``17(``arr``[17]), .``port``18(``arr``[18]), .``port``19(``arr``[19]), .``port``20(``arr``[20]), .``port``21(``arr``[21]), .``port``22(``arr``[22]), .``port``23(``arr``[23]),\
  .``port``24(``arr``[24]), .``port``25(``arr``[25]), .``port``26(``arr``[26]), .``port``27(``arr``[27]), .``port``28(``arr``[28]), .``port``29(``arr``[29]), .``port``30(``arr``[30]), .``port``31(``arr``[31]),\
  .``port``32(``arr``[32]), .``port``33(``arr``[33]), .``port``34(``arr``[34]), .``port``35(``arr``[35]), .``port``36(``arr``[36]), .``port``37(``arr``[37]), .``port``38(``arr``[38]), .``port``39(``arr``[39]),\
  .``port``40(``arr``[40]), .``port``41(``arr``[41]), .``port``42(``arr``[42]), .``port``43(``arr``[43]), .``port``44(``arr``[44]), .``port``45(``arr``[45]), .``port``46(``arr``[46]), .``port``47(``arr``[47]),\
  .``port``48(``arr``[48]), .``port``49(``arr``[49]), .``port``50(``arr``[50]), .``port``51(``arr``[51]), .``port``52(``arr``[52]), .``port``53(``arr``[53]), .``port``54(``arr``[54]), .``port``55(``arr``[55]),\
  .``port``56(``arr``[56]), .``port``57(``arr``[57]), .``port``58(``arr``[58]), .``port``59(``arr``[59]), .``port``60(``arr``[60]), .``port``61(``arr``[61]), .``port``62(``arr``[62]), .``port``63(``arr``[63])

module IDCT_test4();

  localparam CIRCUIT_LATENCY = 29;

  reg signed [15:0] x [63:0];
  reg signed [15:0] out [63:0];
  reg signed [15:0] x_buffer [63:0];
  reg signed [15:0] expected [63:0];
  reg clk;

  IDCT inst (
    `ARRAY_BINDING(x, x),
    `ARRAY_BINDING(out, out),
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
    $display("[IDCT: test 4] Input ready.");
    for (j = 0; j < 64; j = j + 1) begin
      x[j] = x_buffer[j];

      $display("Input[%0d]: [%0d]", j, x[j]);

    end
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("IDCT_test4.vcd");
    $dumpvars(0, IDCT_test4);
    $display("[IDCT: test 4] Started...");

    for (k = 0; k < 64; k = k + 1) begin
      $display("Output[%0d]: %0d", k, out[k]);
      if (expected[k] == out[k]) begin
        $display("GOOD: %0d == %0d", expected[k], out[k]);
      end else begin
        $display("BAD: %0d != %0d", expected[k], out[k]);
        $display("[IDCT: test 4] Stopped.");
        $finish;
      end
    end

    $display("[IDCT: test 4] Stopped.");
    $finish;
  end

endmodule
