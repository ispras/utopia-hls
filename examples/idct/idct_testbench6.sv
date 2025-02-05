// Addition (integer): 1 stage each.
// Subtraction (integer): 1 stage each.
// Multiplication (integer): 3 stages each.
// "Greater" comparison (integer): 3 stages each.
// "Greater or equal" comparison (integer): 3 stages each.
// Total: 29 stages.

`timescale 1s/1s

`define INPUT_INIT(arr)\
  arr[00]=-16'd240; arr[01]=16'd8; arr[02]=-16'd11; arr[03]=16'd47; arr[04]=16'd26; arr[05]=-16'd6; arr[06]=16'd0; arr[07]=16'd5;\
  arr[08]=16'd28; arr[09]=-16'd6; arr[10]=16'd85; arr[11]=16'd44; arr[12]=-16'd4; arr[13]=-16'd25; arr[14]=16'd5; arr[15]=16'd16;\
  arr[16]=16'd21; arr[17]=16'd8; arr[18]=16'd32; arr[19]=-16'd16; arr[20]=-16'd24; arr[21]=16'd0; arr[22]=16'd30; arr[23]=16'd12;\
  arr[24]=-16'd2; arr[25]=16'd18; arr[26]=16'd0; arr[27]=-16'd2; arr[28]=16'd0; arr[29]=16'd7; arr[30]=16'd0; arr[31]=-16'd15;\
  arr[32]=16'd7; arr[33]=16'd4; arr[34]=16'd15; arr[35]=-16'd24; arr[36]=16'd0; arr[37]=16'd9; arr[38]=16'd8; arr[39]=-16'd6;\
  arr[40]=16'd4; arr[41]=16'd9; arr[42]=16'd0; arr[43]=-16'd5; arr[44]=-16'd6; arr[45]=16'd0; arr[46]=16'd0; arr[47]=16'd0;\
  arr[48]=-16'd4; arr[49]=16'd0; arr[50]=-16'd6; arr[51]=16'd0; arr[52]=16'd0; arr[53]=16'd10; arr[54]=-16'd10; arr[55]=-16'd8;\
  arr[56]=16'd6; arr[57]=16'd0; arr[58]=16'd0; arr[59]=16'd0; arr[60]=16'd0; arr[61]=16'd0; arr[62]=16'd0; arr[63]=-16'd8

`define REF_INIT(arr)\
  arr[00]=16'd21; arr[01]=-16'd10; arr[02]=-16'd26; arr[03]=-16'd61; arr[04]=-16'd43; arr[05]=-16'd17; arr[06]=-16'd22; arr[07]=-16'd8;\
  arr[08]=16'd5; arr[09]=-16'd28; arr[10]=-16'd47; arr[11]=-16'd73; arr[12]=-16'd11; arr[13]=-16'd14; arr[14]=-16'd24; arr[15]=-16'd17;\
  arr[16]=-16'd14; arr[17]=-16'd31; arr[18]=-16'd61; arr[19]=-16'd45; arr[20]=-16'd5; arr[21]=-16'd18; arr[22]=-16'd22; arr[23]=-16'd34;\
  arr[24]=-16'd23; arr[25]=-16'd36; arr[26]=-16'd49; arr[27]=-16'd32; arr[28]=-16'd12; arr[29]=-16'd33; arr[30]=-16'd33; arr[31]=-16'd35;\
  arr[32]=-16'd30; arr[33]=-16'd39; arr[34]=-16'd53; arr[35]=-16'd8; arr[36]=-16'd19; arr[37]=-16'd31; arr[38]=-16'd43; arr[39]=-16'd42;\
  arr[40]=-16'd41; arr[41]=-16'd43; arr[42]=-16'd50; arr[43]=-16'd4; arr[44]=-16'd15; arr[45]=-16'd33; arr[46]=-16'd44; arr[47]=-16'd66;\
  arr[48]=-16'd40; arr[49]=-16'd38; arr[50]=-16'd21; arr[51]=-16'd14; arr[52]=-16'd17; arr[53]=-16'd26; arr[54]=-16'd46; arr[55]=-16'd52;\
  arr[56]=-16'd44; arr[57]=-16'd47; arr[58]=-16'd9; arr[59]=-16'd12; arr[60]=-16'd30; arr[61]=-16'd33; arr[62]=-16'd38; arr[63]=-16'd37

`define ARRAY_BINDING(port, arr)\
  .``port``0(``arr``[00]), .``port``1(``arr``[01]), .``port``2(``arr``[02]), .``port``3(``arr``[03]), .``port``4(``arr``[04]), .``port``5(``arr``[05]), .``port``6(``arr``[06]), .``port``7(``arr``[07]),\
  .``port``8(``arr``[08]), .``port``9(``arr``[09]), .``port``10(``arr``[10]), .``port``11(``arr``[11]), .``port``12(``arr``[12]), .``port``13(``arr``[13]), .``port``14(``arr``[14]), .``port``15(``arr``[15]),\
  .``port``16(``arr``[16]), .``port``17(``arr``[17]), .``port``18(``arr``[18]), .``port``19(``arr``[19]), .``port``20(``arr``[20]), .``port``21(``arr``[21]), .``port``22(``arr``[22]), .``port``23(``arr``[23]),\
  .``port``24(``arr``[24]), .``port``25(``arr``[25]), .``port``26(``arr``[26]), .``port``27(``arr``[27]), .``port``28(``arr``[28]), .``port``29(``arr``[29]), .``port``30(``arr``[30]), .``port``31(``arr``[31]),\
  .``port``32(``arr``[32]), .``port``33(``arr``[33]), .``port``34(``arr``[34]), .``port``35(``arr``[35]), .``port``36(``arr``[36]), .``port``37(``arr``[37]), .``port``38(``arr``[38]), .``port``39(``arr``[39]),\
  .``port``40(``arr``[40]), .``port``41(``arr``[41]), .``port``42(``arr``[42]), .``port``43(``arr``[43]), .``port``44(``arr``[44]), .``port``45(``arr``[45]), .``port``46(``arr``[46]), .``port``47(``arr``[47]),\
  .``port``48(``arr``[48]), .``port``49(``arr``[49]), .``port``50(``arr``[50]), .``port``51(``arr``[51]), .``port``52(``arr``[52]), .``port``53(``arr``[53]), .``port``54(``arr``[54]), .``port``55(``arr``[55]),\
  .``port``56(``arr``[56]), .``port``57(``arr``[57]), .``port``58(``arr``[58]), .``port``59(``arr``[59]), .``port``60(``arr``[60]), .``port``61(``arr``[61]), .``port``62(``arr``[62]), .``port``63(``arr``[63])

module IDCT_test6();

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
    $display("[IDCT: test 6] Input ready.");
    for (j = 0; j < 64; j = j + 1) begin
      x[j] = x_buffer[j];

      $display("Input[%0d]: [%0d]", j, x[j]);

    end
  end

  initial begin
    // Wait for the first output.
    #(2*CIRCUIT_LATENCY+3);

    $dumpfile("IDCT_test6.vcd");
    $dumpvars(0, IDCT_test6);
    $display("[IDCT: test 6] Started...");

    for (k = 0; k < 64; k = k + 1) begin
      $display("Output[%0d]: %0d", k, out[k]);
      if (expected[k] == out[k]) begin
        $display("GOOD: %0d == %0d", expected[k], out[k]);
      end else begin
        $display("BAD: %0d != %0d", expected[k], out[k]);
        $display("[IDCT: test 6] Stopped.");
        $finish;
      end
    end

    $display("[IDCT: test 6] Stopped.");
    $finish;
  end

endmodule
