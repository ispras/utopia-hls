module VectorSum(	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:11:5
  input         clock,
                reset,
  input  [15:0] source__dfc_wire_4825__dfc_wire_4825,
                source__dfc_wire_4826__dfc_wire_4825,
                source__dfc_wire_4830__dfc_wire_4825,
                source__dfc_wire_4824__dfc_wire_4825,
                source__dfc_wire_4827__dfc_wire_4825,
                source__dfc_wire_4831__dfc_wire_4825,
                source__dfc_wire_4828__dfc_wire_4825,
                source__dfc_wire_4829__dfc_wire_4825,
  output [15:0] sink__dfc_wire_4848__dfc_wire_4848,
                sink__dfc_wire_4840__dfc_wire_4848,
                sink__dfc_wire_4844__dfc_wire_4848,
                sink__dfc_wire_4836__dfc_wire_4848);

  wire [15:0] _delay_fixed_16_0_1_21_19_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:37:134
  wire [15:0] _delay_fixed_16_0_1_2_18_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:36:130
  wire [15:0] _delay_fixed_16_0_1_39_17_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:35:134
  wire [15:0] _delay_fixed_16_0_1_49_16_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:34:134

  ADD_2x1 ADD2150 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:30:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4824 (_delay_fixed_16_0_1_21_19_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:37:134
    ._dfc_wire_4828 (source__dfc_wire_4828__dfc_wire_4825),
    ._dfc_wire_4836 (sink__dfc_wire_4836__dfc_wire_4848)
  );
  ADD_2x1 ADD2151 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:31:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4824 (_delay_fixed_16_0_1_2_18_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:36:130
    ._dfc_wire_4828 (source__dfc_wire_4829__dfc_wire_4825),
    ._dfc_wire_4836 (sink__dfc_wire_4840__dfc_wire_4848)
  );
  ADD_2x1 ADD2152 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:32:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4824 (_delay_fixed_16_0_1_39_17_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:35:134
    ._dfc_wire_4828 (source__dfc_wire_4830__dfc_wire_4825),
    ._dfc_wire_4836 (sink__dfc_wire_4844__dfc_wire_4848)
  );
  ADD_2x1 ADD2153 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:33:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4824 (source__dfc_wire_4827__dfc_wire_4825),
    ._dfc_wire_4828 (_delay_fixed_16_0_1_49_16_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:34:134
    ._dfc_wire_4836 (sink__dfc_wire_4848__dfc_wire_4848)
  );
  delay_fixed_16_0_1_49 delay_fixed_16_0_1_49_16 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:34:134
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4831__dfc_wire_4825),
    .out   (_delay_fixed_16_0_1_49_16_out)
  );
  delay_fixed_16_0_1_39 delay_fixed_16_0_1_39_17 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:35:134
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4826__dfc_wire_4825),
    .out   (_delay_fixed_16_0_1_39_17_out)
  );
  delay_fixed_16_0_1_2 delay_fixed_16_0_1_2_18 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:36:130
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4825__dfc_wire_4825),
    .out   (_delay_fixed_16_0_1_2_18_out)
  );
  delay_fixed_16_0_1_21 delay_fixed_16_0_1_21_19 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/vector_sum/vectorSumFir.mlir:37:134
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4824__dfc_wire_4825),
    .out   (_delay_fixed_16_0_1_21_19_out)
  );
endmodule

