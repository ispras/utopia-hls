module main(	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:11:5
  input  clock,
         reset,
         n1_x,
         n1_y,
  output n6_z);

  wire _delay_Y_33_8_out;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:27:86
  wire _delay_X_7_7_out;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:26:82
  wire _delay_Z_32_6_out;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:25:86
  wire _n4_z;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:23:50
  wire _n3_z;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:22:57
  wire _n3_w;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:22:57
  wire _n2_x1;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:21:52
  wire _n2_x2;	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:21:52

  split n2 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:21:52
    .clock (clock),
    .reset (reset),
    .x     (n1_x),
    .x1    (_n2_x1),
    .x2    (_n2_x2)
  );
  kernel1 n3 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:22:57
    .clock (clock),
    .reset (reset),
    .x     (_n2_x1),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:21:52
    .y     (_delay_Y_33_8_out),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:27:86
    .z     (_n3_z),
    .w     (_n3_w)
  );
  kernel2 n4 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:23:50
    .clock (clock),
    .reset (reset),
    .x     (_delay_X_7_7_out),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:26:82
    .w     (_n3_w),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:22:57
    .z     (_n4_z)
  );
  merge n5 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:24:52
    .clock (clock),
    .reset (reset),
    .z1    (_delay_Z_32_6_out),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:25:86
    .z2    (_n4_z),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:23:50
    .z     (n6_z)
  );
  delay_Z_32 delay_Z_32_6 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:25:86
    .clock (clock),
    .reset (reset),
    .in    (_n3_z),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:22:57
    .out   (_delay_Z_32_6_out)
  );
  delay_X_7 delay_X_7_7 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:26:82
    .clock (clock),
    .reset (reset),
    .in    (_n2_x2),	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:21:52
    .out   (_delay_X_7_7_out)
  );
  delay_Y_33 delay_Y_33_8 (	// /home/nikita/Desktop/work/utopia/output/test/hil/test/testFir.mlir:27:86
    .clock (clock),
    .reset (reset),
    .in    (n1_y),
    .out   (_delay_Y_33_8_out)
  );
endmodule

