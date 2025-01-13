module {
  dfcir.kernel "Scalar3" {
    %0 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y3")
    %1 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x3")
    %2 = dfcir.mul[?] (%1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %3 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y2")
    %4 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x2")
    %5 = dfcir.mul[?] (%4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %3 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %6 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("y1")
    %7 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x1")
    %8 = dfcir.mul[?] (%7 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %6 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %9 = dfcir.add[?] (%8 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %5 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %10 = dfcir.add[?] (%9 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %2 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %11 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%11 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %10 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
  }
}
