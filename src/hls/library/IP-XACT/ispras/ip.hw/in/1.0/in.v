module IN(

    clock,
    reset,
    x);

output [15:0] x;
input clock;
input reset;
wire [15:0] x;
reg [15:0] state_0;
assign x = state_0[15:0];
endmodule //IN
