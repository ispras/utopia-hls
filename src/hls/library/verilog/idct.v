
// Evaluations of rows
module idctrow(input clock, input reset, input [5:0] start_index);

reg [15:0] x0;

always @(posedge clock) begin
  if(!reset) begin
    x0 <= 0;
  end else begin
    x0 <= start_index;
  end
end


//  integer i;
//  always @*
//    for (i = 0; i < 64; i = i + 1)
//      block_array[i] = block_flat[16 * i +: 16];

endmodule // idctrow


// Evaluations of columns
module idctcol(input clock, input reset, input [5:0] start_index);

reg [15:0] x0;
reg [15:0] x1;
reg [15:0] x2;
reg [15:0] x3;
reg [15:0] x4;
reg [15:0] x5;
reg [15:0] x6;
reg [15:0] x7;
reg [15:0] x8;

always @(posedge clock) begin
  if (!reset) begin
    x0 <= 0;
    x1 <= 0;
    x2 <= 0;
    x3 <= 0;
    x4 <= 0;
    x5 <= 0;
    x6 <= 0;
    x7 <= 0;
    x8 <= 0;
  end else begin
    x1 <= start_index;
  end
end

endmodule // idctcol


module Fast_IDCT(input clock, input reset);
  wire [5:0] start_index;
  genvar i;

  generate for (i = 0; i < 8; i = i + 1) begin
    idctrow ir0 (clock, reset, i == 0 ? 6'd0 :
                               i == 1 ? 6'd8 :
                               i == 2 ? 6'd16 :
                               i == 3 ? 6'd24 :
                               i == 4 ? 6'd32 :
                               i == 5 ? 6'd40 :
                               i == 6 ? 6'd48 :
                               i == 7 ? 6'd56 : 6'd0 /* start_index */);
  end
  endgenerate

  generate for (i = 0; i < 8; i = i + 1) begin
    idctcol ic0 (clock, reset, i == 0 ? 6'd0 :
                               i == 1 ? 6'd1 :
                               i == 2 ? 6'd2 :
                               i == 3 ? 6'd3 :
                               i == 4 ? 6'd4 :
                               i == 5 ? 6'd5 :
                               i == 6 ? 6'd6 :
                               i == 7 ? 6'd7 : 6'd0 /* start_index */);
  end
  endgenerate
endmodule // Fast_IDCT

module top (input clock, input reset);
reg [15:0] block [63:0]; // TODO: how to pass that? by means of flattening?
Fast_IDCT idct(clock, reset);

endmodule // top
