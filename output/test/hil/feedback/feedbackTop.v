module main(	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:11:5
  input  clock,
         reset,
         n1_x,
  output n4_w);

  wire _n3_y;	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:21:50
  wire _n2_z;	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:20:50

  kernel1 n2 (	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:20:50
    .clock (clock),
    .reset (reset),
    .x     (n1_x),
    .y     (_n3_y),	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:21:50
    .z     (_n2_z)
  );
  kernel2 n3 (	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:21:50
    .clock (clock),
    .reset (reset),
    .z     (_n2_z),	// /home/nikita/Desktop/work/utopia/output/test/hil/feedback/feedbackFir.mlir:20:50
    .w     (n4_w),
    .y     (_n3_y)
  );
endmodule

