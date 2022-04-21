module SHR_14(

    clock,
    reset,
    x,
    z);

input [15:0] x;
output [15:0] z;
input clock;
input reset;
wire [15:0] x;
wire [15:0] z;
assign z = x >> 14;
endmodule //SHR_14
