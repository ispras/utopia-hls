module add (clock, reset, x, y, z, c);

parameter WIDTH = 8;

input clock;
input reset;
input [WIDTH - 1 : 0] x;
input [WIDTH - 1 : 0] y;
output [WIDTH - 1 : 0] z;
output c;
reg [WIDTH - 1 : 0] z;
reg c;

always @(posedge clock) begin
  if (!reset) begin
    {c, z} <= x + y;
  end
  else begin
    {c, z} <= 0;
  end
end

endmodule // add
