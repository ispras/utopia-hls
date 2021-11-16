module clip (clock, reset, x, z);

parameter WIDTH = 8;
parameter LEVEL = 4;

input clock;
input reset;
input [WIDTH - 1 : 0] x;
output [WIDTH - 1 : 0] z;
reg [WIDTH - 1 : 0] z;

always @(posedge clock) begin
  if (!reset) begin
    z <= x < -LEVEL && x > {(WIDTH - 1){1'b1}} ? -LEVEL :
                                                 x >= LEVEL ? (LEVEL - 1) : x;
  end
  else begin
    z <= 0;
  end
end

endmodule // clip
