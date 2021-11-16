module mul (clock, reset, x, y, z);

parameter WIDTH = 8;

input clock;
input reset;
input [WIDTH - 1 : 0] x;
input [WIDTH - 1 : 0] y;
output [2 * WIDTH - 1 : 0] z;
reg [2 * WIDTH - 1 : 0] z;

always @(posedge clock) begin
  if (!reset) begin
    z <= x * y;
  end
  else begin
    z <= 0;
  end
end

endmodule // mul
