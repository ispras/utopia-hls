module ScalarMul(	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:11:5
  input         clock,
                reset,
  input  [15:0] source__dfc_wire_4853__dfc_wire_4853,
                source__dfc_wire_4857__dfc_wire_4853,
                source__dfc_wire_4854__dfc_wire_4853,
                source__dfc_wire_4855__dfc_wire_4853,
                source__dfc_wire_4856__dfc_wire_4853,
                source__dfc_wire_4852__dfc_wire_4853,
  output [15:0] sink__dfc_wire_4873__dfc_wire_4873);

  wire [15:0] _delay_fixed_16_0_1_34_16_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:34:134
  wire [15:0] _delay_fixed_16_0_1_22_15_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:33:134
  wire [15:0] _delay_fixed_16_0_1_11_14_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:32:134
  wire [15:0] _delay_fixed_16_0_1_25_13_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:31:134
  wire [15:0] _delay_fixed_16_0_1_34_12_out;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:30:134
  wire [15:0] _MUL2169__dfc_wire_4859;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:28:114
  wire [15:0] _ADD2168__dfc_wire_4866;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:27:114
  wire [15:0] _MUL2167__dfc_wire_4859;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:26:114
  wire [15:0] _MUL2166__dfc_wire_4859;	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:25:114

  MUL_2x1 MUL2166 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:25:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4852 (_delay_fixed_16_0_1_34_16_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:34:134
    ._dfc_wire_4855 (source__dfc_wire_4855__dfc_wire_4853),
    ._dfc_wire_4859 (_MUL2166__dfc_wire_4859)
  );
  MUL_2x1 MUL2167 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:26:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4852 (_delay_fixed_16_0_1_22_15_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:33:134
    ._dfc_wire_4855 (source__dfc_wire_4856__dfc_wire_4853),
    ._dfc_wire_4859 (_MUL2167__dfc_wire_4859)
  );
  ADD_2x1 ADD2168 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:27:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4859 (_delay_fixed_16_0_1_11_14_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:32:134
    ._dfc_wire_4863 (_MUL2167__dfc_wire_4859),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:26:114
    ._dfc_wire_4866 (_ADD2168__dfc_wire_4866)
  );
  MUL_2x1 MUL2169 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:28:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4852 (_delay_fixed_16_0_1_25_13_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:31:134
    ._dfc_wire_4855 (source__dfc_wire_4857__dfc_wire_4853),
    ._dfc_wire_4859 (_MUL2169__dfc_wire_4859)
  );
  ADD_2x1 ADD2170 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:29:114
    .clock          (clock),
    .reset          (reset),
    ._dfc_wire_4859 (_ADD2168__dfc_wire_4866),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:27:114
    ._dfc_wire_4863 (_delay_fixed_16_0_1_34_12_out),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:30:134
    ._dfc_wire_4866 (sink__dfc_wire_4873__dfc_wire_4873)
  );
  delay_fixed_16_0_1_34 delay_fixed_16_0_1_34_12 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:30:134
    .clock (clock),
    .reset (reset),
    .in    (_MUL2169__dfc_wire_4859),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:28:114
    .out   (_delay_fixed_16_0_1_34_12_out)
  );
  delay_fixed_16_0_1_25 delay_fixed_16_0_1_25_13 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:31:134
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4854__dfc_wire_4853),
    .out   (_delay_fixed_16_0_1_25_13_out)
  );
  delay_fixed_16_0_1_11 delay_fixed_16_0_1_11_14 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:32:134
    .clock (clock),
    .reset (reset),
    .in    (_MUL2166__dfc_wire_4859),	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:25:114
    .out   (_delay_fixed_16_0_1_11_14_out)
  );
  delay_fixed_16_0_1_22 delay_fixed_16_0_1_22_15 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:33:134
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4853__dfc_wire_4853),
    .out   (_delay_fixed_16_0_1_22_15_out)
  );
  delay_fixed_16_0_1_34 delay_fixed_16_0_1_34_16 (	// /home/nikita/Desktop/work/utopia/output/test/dfc/scalar_mul/scalarMulFir.mlir:34:134
    .clock (clock),
    .reset (reset),
    .in    (source__dfc_wire_4852__dfc_wire_4853),
    .out   (_delay_fixed_16_0_1_34_16_out)
  );
endmodule

