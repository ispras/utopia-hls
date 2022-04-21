module dup_2(

    clock,
    reset,
    x,
    y,
    z);

input [15:0] x;
output [15:0] y;
output [15:0] z;
input clock;
input reset;
wire [15:0] x;
wire [15:0] y;
wire [15:0] z;
assign y = x;
assign z = x;
endmodule //dup_2
