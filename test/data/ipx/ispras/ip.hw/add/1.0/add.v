module ADD(

    clock,
    reset,
    x,
    y,
    z);

input [15:0] x;
input [15:0] y;
output [15:0] z;
input clock;
input reset;
wire [15:0] x;
wire [15:0] y;
wire [15:0] z;
assign {z} = x + y;
endmodule //ADD
