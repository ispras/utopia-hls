module delay_INT16_4(

    clock,
    reset,
    in,
    out);

input [15:0] in;
output [15:0] out;
input clock;
input reset;
wire [15:0] in;
wire [15:0] out;
reg [15:0] state_0;
reg [15:0] state_1;
reg [15:0] state_2;
always @(posedge clock) begin
state_0 <= {in};
state_1 <= {state_0[14:0], state_0[15]};
state_2 <= {state_1[14:0], state_1[15]};
end
assign out = state_3[15:0];
endmodule //delay_INT16_4
