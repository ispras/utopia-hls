module shr (clock, reset, x, z);

parameter WIDTH = 8;
parameter SHIFT = 1;

input clock;
input reset;
input [WIDTH - 1 : 0] x;
output [WIDTH - 1 : 0] z;
reg [WIDTH - 1 : 0] z;

always @(posedge clock) begin
  if (!reset) begin
    z <= x >> SHIFT;
  end
  else begin
    z <= 0;
  end
end

endmodule // shr
