/*
 * This Verilog testbench was automatically generated by Utopia EDA
 * Generation started: 7-2-2023 0:51:29
 *
 * Ivannikov Institute for System Programming
 * of the Russian Academy of Sciences (ISP RAS)
 * 25 Alexander Solzhenitsyn st., Moscow, 109004, Russia
 * http://forge.ispras.ru/projects/utopia
 */

`timescale 1ns/10ps

module main_tb;
	// inputs
	reg  clock;
	reg  reset;
	reg [0:0] n_in_0_0_x;
	reg [0:0] n_in_0_1_x;
	reg [0:0] n_in_0_2_x;
	reg [0:0] n_in_0_3_x;
	reg [0:0] n_in_0_4_x;
	reg [0:0] n_in_0_5_x;
	reg [0:0] n_in_0_6_x;
	reg [0:0] n_in_0_7_x;
	reg [0:0] n_in_1_0_x;
	reg [0:0] n_in_1_1_x;
	reg [0:0] n_in_1_2_x;
	reg [0:0] n_in_1_3_x;
	reg [0:0] n_in_1_4_x;
	reg [0:0] n_in_1_5_x;
	reg [0:0] n_in_1_6_x;
	reg [0:0] n_in_1_7_x;
	reg [0:0] n_in_2_0_x;
	reg [0:0] n_in_2_1_x;
	reg [0:0] n_in_2_2_x;
	reg [0:0] n_in_2_3_x;
	reg [0:0] n_in_2_4_x;
	reg [0:0] n_in_2_5_x;
	reg [0:0] n_in_2_6_x;
	reg [0:0] n_in_2_7_x;
	reg [0:0] n_in_3_0_x;
	reg [0:0] n_in_3_1_x;
	reg [0:0] n_in_3_2_x;
	reg [0:0] n_in_3_3_x;
	reg [0:0] n_in_3_4_x;
	reg [0:0] n_in_3_5_x;
	reg [0:0] n_in_3_6_x;
	reg [0:0] n_in_3_7_x;
	reg [0:0] n_in_4_0_x;
	reg [0:0] n_in_4_1_x;
	reg [0:0] n_in_4_2_x;
	reg [0:0] n_in_4_3_x;
	reg [0:0] n_in_4_4_x;
	reg [0:0] n_in_4_5_x;
	reg [0:0] n_in_4_6_x;
	reg [0:0] n_in_4_7_x;
	reg [0:0] n_in_5_0_x;
	reg [0:0] n_in_5_1_x;
	reg [0:0] n_in_5_2_x;
	reg [0:0] n_in_5_3_x;
	reg [0:0] n_in_5_4_x;
	reg [0:0] n_in_5_5_x;
	reg [0:0] n_in_5_6_x;
	reg [0:0] n_in_5_7_x;
	reg [0:0] n_in_6_0_x;
	reg [0:0] n_in_6_1_x;
	reg [0:0] n_in_6_2_x;
	reg [0:0] n_in_6_3_x;
	reg [0:0] n_in_6_4_x;
	reg [0:0] n_in_6_5_x;
	reg [0:0] n_in_6_6_x;
	reg [0:0] n_in_6_7_x;
	reg [0:0] n_in_7_0_x;
	reg [0:0] n_in_7_1_x;
	reg [0:0] n_in_7_2_x;
	reg [0:0] n_in_7_3_x;
	reg [0:0] n_in_7_4_x;
	reg [0:0] n_in_7_5_x;
	reg [0:0] n_in_7_6_x;
	reg [0:0] n_in_7_7_x;

	// outputs
	wire [0:0] n_out_0_0_x;
	wire [0:0] n_out_0_1_x;
	wire [0:0] n_out_0_2_x;
	wire [0:0] n_out_0_3_x;
	wire [0:0] n_out_0_4_x;
	wire [0:0] n_out_0_5_x;
	wire [0:0] n_out_0_6_x;
	wire [0:0] n_out_0_7_x;
	wire [0:0] n_out_1_0_x;
	wire [0:0] n_out_1_1_x;
	wire [0:0] n_out_1_2_x;
	wire [0:0] n_out_1_3_x;
	wire [0:0] n_out_1_4_x;
	wire [0:0] n_out_1_5_x;
	wire [0:0] n_out_1_6_x;
	wire [0:0] n_out_1_7_x;
	wire [0:0] n_out_2_0_x;
	wire [0:0] n_out_2_1_x;
	wire [0:0] n_out_2_2_x;
	wire [0:0] n_out_2_3_x;
	wire [0:0] n_out_2_4_x;
	wire [0:0] n_out_2_5_x;
	wire [0:0] n_out_2_6_x;
	wire [0:0] n_out_2_7_x;
	wire [0:0] n_out_3_0_x;
	wire [0:0] n_out_3_1_x;
	wire [0:0] n_out_3_2_x;
	wire [0:0] n_out_3_3_x;
	wire [0:0] n_out_3_4_x;
	wire [0:0] n_out_3_5_x;
	wire [0:0] n_out_3_6_x;
	wire [0:0] n_out_3_7_x;
	wire [0:0] n_out_4_0_x;
	wire [0:0] n_out_4_1_x;
	wire [0:0] n_out_4_2_x;
	wire [0:0] n_out_4_3_x;
	wire [0:0] n_out_4_4_x;
	wire [0:0] n_out_4_5_x;
	wire [0:0] n_out_4_6_x;
	wire [0:0] n_out_4_7_x;
	wire [0:0] n_out_5_0_x;
	wire [0:0] n_out_5_1_x;
	wire [0:0] n_out_5_2_x;
	wire [0:0] n_out_5_3_x;
	wire [0:0] n_out_5_4_x;
	wire [0:0] n_out_5_5_x;
	wire [0:0] n_out_5_6_x;
	wire [0:0] n_out_5_7_x;
	wire [0:0] n_out_6_0_x;
	wire [0:0] n_out_6_1_x;
	wire [0:0] n_out_6_2_x;
	wire [0:0] n_out_6_3_x;
	wire [0:0] n_out_6_4_x;
	wire [0:0] n_out_6_5_x;
	wire [0:0] n_out_6_6_x;
	wire [0:0] n_out_6_7_x;
	wire [0:0] n_out_7_0_x;
	wire [0:0] n_out_7_1_x;
	wire [0:0] n_out_7_2_x;
	wire [0:0] n_out_7_3_x;
	wire [0:0] n_out_7_4_x;
	wire [0:0] n_out_7_5_x;
	wire [0:0] n_out_7_6_x;
	wire [0:0] n_out_7_7_x;

	localparam main_latency = 1927;

	main DUT(
		.clock(clock),
		.reset(reset),
		.n_in_0_0_x(n_in_0_0_x),
		.n_in_0_1_x(n_in_0_1_x),
		.n_in_0_2_x(n_in_0_2_x),
		.n_in_0_3_x(n_in_0_3_x),
		.n_in_0_4_x(n_in_0_4_x),
		.n_in_0_5_x(n_in_0_5_x),
		.n_in_0_6_x(n_in_0_6_x),
		.n_in_0_7_x(n_in_0_7_x),
		.n_in_1_0_x(n_in_1_0_x),
		.n_in_1_1_x(n_in_1_1_x),
		.n_in_1_2_x(n_in_1_2_x),
		.n_in_1_3_x(n_in_1_3_x),
		.n_in_1_4_x(n_in_1_4_x),
		.n_in_1_5_x(n_in_1_5_x),
		.n_in_1_6_x(n_in_1_6_x),
		.n_in_1_7_x(n_in_1_7_x),
		.n_in_2_0_x(n_in_2_0_x),
		.n_in_2_1_x(n_in_2_1_x),
		.n_in_2_2_x(n_in_2_2_x),
		.n_in_2_3_x(n_in_2_3_x),
		.n_in_2_4_x(n_in_2_4_x),
		.n_in_2_5_x(n_in_2_5_x),
		.n_in_2_6_x(n_in_2_6_x),
		.n_in_2_7_x(n_in_2_7_x),
		.n_in_3_0_x(n_in_3_0_x),
		.n_in_3_1_x(n_in_3_1_x),
		.n_in_3_2_x(n_in_3_2_x),
		.n_in_3_3_x(n_in_3_3_x),
		.n_in_3_4_x(n_in_3_4_x),
		.n_in_3_5_x(n_in_3_5_x),
		.n_in_3_6_x(n_in_3_6_x),
		.n_in_3_7_x(n_in_3_7_x),
		.n_in_4_0_x(n_in_4_0_x),
		.n_in_4_1_x(n_in_4_1_x),
		.n_in_4_2_x(n_in_4_2_x),
		.n_in_4_3_x(n_in_4_3_x),
		.n_in_4_4_x(n_in_4_4_x),
		.n_in_4_5_x(n_in_4_5_x),
		.n_in_4_6_x(n_in_4_6_x),
		.n_in_4_7_x(n_in_4_7_x),
		.n_in_5_0_x(n_in_5_0_x),
		.n_in_5_1_x(n_in_5_1_x),
		.n_in_5_2_x(n_in_5_2_x),
		.n_in_5_3_x(n_in_5_3_x),
		.n_in_5_4_x(n_in_5_4_x),
		.n_in_5_5_x(n_in_5_5_x),
		.n_in_5_6_x(n_in_5_6_x),
		.n_in_5_7_x(n_in_5_7_x),
		.n_in_6_0_x(n_in_6_0_x),
		.n_in_6_1_x(n_in_6_1_x),
		.n_in_6_2_x(n_in_6_2_x),
		.n_in_6_3_x(n_in_6_3_x),
		.n_in_6_4_x(n_in_6_4_x),
		.n_in_6_5_x(n_in_6_5_x),
		.n_in_6_6_x(n_in_6_6_x),
		.n_in_6_7_x(n_in_6_7_x),
		.n_in_7_0_x(n_in_7_0_x),
		.n_in_7_1_x(n_in_7_1_x),
		.n_in_7_2_x(n_in_7_2_x),
		.n_in_7_3_x(n_in_7_3_x),
		.n_in_7_4_x(n_in_7_4_x),
		.n_in_7_5_x(n_in_7_5_x),
		.n_in_7_6_x(n_in_7_6_x),
		.n_in_7_7_x(n_in_7_7_x),
		.n_out_0_0_x(n_out_0_0_x),
		.n_out_0_1_x(n_out_0_1_x),
		.n_out_0_2_x(n_out_0_2_x),
		.n_out_0_3_x(n_out_0_3_x),
		.n_out_0_4_x(n_out_0_4_x),
		.n_out_0_5_x(n_out_0_5_x),
		.n_out_0_6_x(n_out_0_6_x),
		.n_out_0_7_x(n_out_0_7_x),
		.n_out_1_0_x(n_out_1_0_x),
		.n_out_1_1_x(n_out_1_1_x),
		.n_out_1_2_x(n_out_1_2_x),
		.n_out_1_3_x(n_out_1_3_x),
		.n_out_1_4_x(n_out_1_4_x),
		.n_out_1_5_x(n_out_1_5_x),
		.n_out_1_6_x(n_out_1_6_x),
		.n_out_1_7_x(n_out_1_7_x),
		.n_out_2_0_x(n_out_2_0_x),
		.n_out_2_1_x(n_out_2_1_x),
		.n_out_2_2_x(n_out_2_2_x),
		.n_out_2_3_x(n_out_2_3_x),
		.n_out_2_4_x(n_out_2_4_x),
		.n_out_2_5_x(n_out_2_5_x),
		.n_out_2_6_x(n_out_2_6_x),
		.n_out_2_7_x(n_out_2_7_x),
		.n_out_3_0_x(n_out_3_0_x),
		.n_out_3_1_x(n_out_3_1_x),
		.n_out_3_2_x(n_out_3_2_x),
		.n_out_3_3_x(n_out_3_3_x),
		.n_out_3_4_x(n_out_3_4_x),
		.n_out_3_5_x(n_out_3_5_x),
		.n_out_3_6_x(n_out_3_6_x),
		.n_out_3_7_x(n_out_3_7_x),
		.n_out_4_0_x(n_out_4_0_x),
		.n_out_4_1_x(n_out_4_1_x),
		.n_out_4_2_x(n_out_4_2_x),
		.n_out_4_3_x(n_out_4_3_x),
		.n_out_4_4_x(n_out_4_4_x),
		.n_out_4_5_x(n_out_4_5_x),
		.n_out_4_6_x(n_out_4_6_x),
		.n_out_4_7_x(n_out_4_7_x),
		.n_out_5_0_x(n_out_5_0_x),
		.n_out_5_1_x(n_out_5_1_x),
		.n_out_5_2_x(n_out_5_2_x),
		.n_out_5_3_x(n_out_5_3_x),
		.n_out_5_4_x(n_out_5_4_x),
		.n_out_5_5_x(n_out_5_5_x),
		.n_out_5_6_x(n_out_5_6_x),
		.n_out_5_7_x(n_out_5_7_x),
		.n_out_6_0_x(n_out_6_0_x),
		.n_out_6_1_x(n_out_6_1_x),
		.n_out_6_2_x(n_out_6_2_x),
		.n_out_6_3_x(n_out_6_3_x),
		.n_out_6_4_x(n_out_6_4_x),
		.n_out_6_5_x(n_out_6_5_x),
		.n_out_6_6_x(n_out_6_6_x),
		.n_out_6_7_x(n_out_6_7_x),
		.n_out_7_0_x(n_out_7_0_x),
		.n_out_7_1_x(n_out_7_1_x),
		.n_out_7_2_x(n_out_7_2_x),
		.n_out_7_3_x(n_out_7_3_x),
		.n_out_7_4_x(n_out_7_4_x),
		.n_out_7_5_x(n_out_7_5_x),
		.n_out_7_6_x(n_out_7_6_x),
		.n_out_7_7_x(n_out_7_7_x));

	initial clock = 0;
	always clock = #(main_latency) ~clock;

	initial begin
		$dumpvars(0, main_tb);

		// reset the device
		@(posedge clock);
		
		reset = 0;
		#(main_latency * 6);
		reset = ~0;
		

		// apply sequence of random stimuli
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		@(posedge clock);
		
		n_in_0_0_x = 0;
		n_in_0_1_x = 0;
		n_in_0_2_x = 0;
		n_in_0_3_x = 0;
		n_in_0_4_x = 0;
		n_in_0_5_x = 0;
		n_in_0_6_x = 0;
		n_in_0_7_x = 0;
		n_in_1_0_x = 0;
		n_in_1_1_x = 0;
		n_in_1_2_x = 0;
		n_in_1_3_x = 0;
		n_in_1_4_x = 0;
		n_in_1_5_x = 0;
		n_in_1_6_x = 0;
		n_in_1_7_x = 0;
		n_in_2_0_x = 0;
		n_in_2_1_x = 0;
		n_in_2_2_x = 0;
		n_in_2_3_x = 0;
		n_in_2_4_x = 0;
		n_in_2_5_x = 0;
		n_in_2_6_x = 0;
		n_in_2_7_x = 0;
		n_in_3_0_x = 0;
		n_in_3_1_x = 0;
		n_in_3_2_x = 0;
		n_in_3_3_x = 0;
		n_in_3_4_x = 0;
		n_in_3_5_x = 0;
		n_in_3_6_x = 0;
		n_in_3_7_x = 0;
		n_in_4_0_x = 0;
		n_in_4_1_x = 0;
		n_in_4_2_x = 0;
		n_in_4_3_x = 0;
		n_in_4_4_x = 0;
		n_in_4_5_x = 0;
		n_in_4_6_x = 0;
		n_in_4_7_x = 0;
		n_in_5_0_x = 0;
		n_in_5_1_x = 0;
		n_in_5_2_x = 0;
		n_in_5_3_x = 0;
		n_in_5_4_x = 0;
		n_in_5_5_x = 0;
		n_in_5_6_x = 0;
		n_in_5_7_x = 0;
		n_in_6_0_x = 0;
		n_in_6_1_x = 0;
		n_in_6_2_x = 0;
		n_in_6_3_x = 0;
		n_in_6_4_x = 0;
		n_in_6_5_x = 0;
		n_in_6_6_x = 0;
		n_in_6_7_x = 0;
		n_in_7_0_x = 0;
		n_in_7_1_x = 0;
		n_in_7_2_x = 0;
		n_in_7_3_x = 0;
		n_in_7_4_x = 0;
		n_in_7_5_x = 0;
		n_in_7_6_x = 0;
		n_in_7_7_x = 0;
		
		#(main_latency * 240);

		$finish;
	end
endmodule
