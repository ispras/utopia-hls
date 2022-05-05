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

module C128(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 128;
endmodule //C128

module C181(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 181;
endmodule //C181

module C4(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 4;
endmodule //C4

module C8192(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 8192;
endmodule //C8192

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

module MUL(

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
assign {z} = x * y;
endmodule //MUL

module OUT(

    clock,
    reset,
    x);

input [15:0] x;
input clock;
input reset;
wire [15:0] x;
endmodule //OUT

module SHL_11(

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
assign z = x << 11;
endmodule //SHL_11

module SHL_8(

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
assign z = x << 8;
endmodule //SHL_8

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

module SHR_3(

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
assign z = x >> 3;
endmodule //SHR_3

module SHR_8(

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
assign z = x >> 8;
endmodule //SHR_8

module SUB(

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
assign {z} = x - y;
endmodule //SUB

module W1(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 2841;
endmodule //W1

module W1_add_W7(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = (2841+565);
endmodule //W1_add_W7

module W1_sub_W7(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = (2841-565);
endmodule //W1_sub_W7

module W2(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 2676;
endmodule //W2

module W2_add_W6(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = (2676+1108);
endmodule //W2_add_W6

module W2_sub_W6(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = (2676-1108);
endmodule //W2_sub_W6

module W3(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 2408;
endmodule //W3

module W3_add_W5(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = (2408+1609);
endmodule //W3_add_W5

module W3_sub_W5(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = (2408-1609);
endmodule //W3_sub_W5

module W5(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 1609;
endmodule //W5

module W6(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 1108;
endmodule //W6

module W7(

    clock,
    reset,
    value);

output [15:0] value;
input clock;
input reset;
wire [15:0] value;
assign value = 565;
endmodule //W7

module delay_INT16_1(

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
assign out = state_0[15:0];
endmodule //delay_INT16_1

module delay_INT16_2(

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
assign out = state_0[15:0];
endmodule //delay_INT16_2

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
assign out = state_0[15:0];
endmodule //delay_INT16_4

module delay_INT16_6(

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
assign out = state_0[15:0];
endmodule //delay_INT16_6

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

