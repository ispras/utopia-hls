/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module delay_X_7(
	// inputs
	clock,
	reset,
	in,

	// outputs
	out
);


input clock;
input reset;
input in;
output out;
wire in;
reg out;
reg [31:0] state;
reg [31:0] s0;
reg [31:0] s1;
reg [31:0] s2;
always @(posedge clock) begin
if (!reset) begin
  state <= 0; end
else if (state == 0) begin
  state <= 1;
  s0 <= in; end
else if (state == 1) begin
  state <= 2;
  s1 <= s0; end
else if (state == 2) begin
  state <= 3;
  s2 <= s1; end
else begin
  state <= 0;
  out <= s2; end
end

endmodule // delay_X_7
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module delay_Y_33(
	// inputs
	clock,
	reset,
	in,

	// outputs
	out
);


input clock;
input reset;
input in;
output out;
wire in;
reg out;
reg [31:0] state;
reg [31:0] s0;
reg [31:0] s1;
reg [31:0] s2;
always @(posedge clock) begin
if (!reset) begin
  state <= 0; end
else if (state == 0) begin
  state <= 1;
  s0 <= in; end
else if (state == 1) begin
  state <= 2;
  s1 <= s0; end
else if (state == 2) begin
  state <= 3;
  s2 <= s1; end
else begin
  state <= 0;
  out <= s2; end
end

endmodule // delay_Y_33
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module delay_Z_32(
	// inputs
	clock,
	reset,
	in,

	// outputs
	out
);


input clock;
input reset;
input in;
output out;
wire in;
reg out;
reg [31:0] state;
reg [31:0] s0;
reg [31:0] s1;
reg [31:0] s2;
always @(posedge clock) begin
if (!reset) begin
  state <= 0; end
else if (state == 0) begin
  state <= 1;
  s0 <= in; end
else if (state == 1) begin
  state <= 2;
  s1 <= s0; end
else if (state == 2) begin
  state <= 3;
  s2 <= s1; end
else begin
  state <= 0;
  out <= s2; end
end

endmodule // delay_Z_32
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module kernel1(
	// inputs
	clock,
	reset,
	x,
	y,

	// outputs
	z,
	w
);


input clock;
input reset;
input x;
input y;
output z;
output w;
wire x;
wire y;
wire z;
wire w;
reg [1:0] state_0;
always @(posedge clock) begin
state_0 <= {x, y};
end
assign z = state_0[0:0];
assign w = state_0[1:1];

endmodule // kernel1
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module kernel2(
	// inputs
	clock,
	reset,
	x,
	w,

	// outputs
	z
);


input clock;
input reset;
input x;
input w;
output z;
wire x;
wire w;
wire z;
reg [1:0] state_0;
always @(posedge clock) begin
state_0 <= {x, w};
end
assign z = state_0[0:0];

endmodule // kernel2
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module merge(
	// inputs
	clock,
	reset,
	z1,
	z2,

	// outputs
	z
);


input clock;
input reset;
input z1;
input z2;
output z;
wire z1;
wire z2;
reg z;
reg [31:0] state;
always @(posedge clock) begin
if (!reset) begin
  state <= 0; end
else if (state == 0) begin
  state <= 1;
  z <= z1; end
else  if (state == 1) begin
  state <= 2;
  z <= z2; end
else begin
  state <= 0; end
end

endmodule // merge
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module sink(
	// inputs
	clock,
	reset,
	z

	// outputs
);


input clock;
input reset;
input z;
wire z;

endmodule // sink
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module source(
	// inputs
	clock,
	reset,

	// outputs
	x,
	y
);


input clock;
input reset;
output x;
output y;
wire x;
wire y;

endmodule // source
/*
 * This Verilog library file was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

module split(
	// inputs
	clock,
	reset,
	x,

	// outputs
	x1,
	x2
);


input clock;
input reset;
input x;
output x1;
output x2;
wire x;
reg x1;
reg x2;
reg [31:0] state;
always @(posedge clock) begin
if (!reset) begin
  state <= 0; end
else if (state == 0) begin
  state <= 1;
  x1 <= x; end
else  if (state == 1) begin
  state <= 2;
  x2 <= x; end
else begin
  state <= 0; end
end

endmodule // split
