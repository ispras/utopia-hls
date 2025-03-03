module {
  dfcir.kernel "MatrixMul2" {
    %0 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y21")
    %1 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y12")
    %2 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y11")
    %3 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x22")
    %4 = dfcir.mul[?] (%3 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %5 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y22")
    %6 = dfcir.mul[?] (%3 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %5 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %7 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x21")
    %8 = dfcir.mul[?] (%7 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %9 = dfcir.add[?] (%8 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %6 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %10 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out22") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%10 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %9 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
    %11 = dfcir.mul[?] (%7 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %2 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %12 = dfcir.add[?] (%11 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %13 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out21") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%13 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %12 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
    %14 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x12")
    %15 = dfcir.mul[?] (%14 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %5 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %16 = dfcir.mul[?] (%14 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %17 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x11")
    %18 = dfcir.mul[?] (%17 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %19 = dfcir.add[?] (%18 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %15 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %20 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out12") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%20 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %19 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
    %21 = dfcir.mul[?] (%17 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %2 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %22 = dfcir.add[?] (%21 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %16 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %23 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out11") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%23 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %22 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
  }
}
