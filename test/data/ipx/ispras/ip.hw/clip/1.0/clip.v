module CLIP(

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
assign z = clip16(x);
function [15:0] clip16;
  input [15:0] in;
  begin
    if (in[15] == 1 && in[14:8] != 7'h7F)
      clip16 = 8'h80;
    else if (in[15] == 0 && in [14:8] != 0)
      clip16 = 8'h7F;
    else
      clip16 = in;
  end
endfunction
endmodule //CLIP
