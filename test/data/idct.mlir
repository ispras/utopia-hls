module {
  dfcir.kernel "IDCT" {
    %0 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %1 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %2 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %3 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %4 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %5 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %6 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %7 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %8 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %9 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %10 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %11 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %12 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %13 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %14 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %15 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %16 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %17 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %18 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %19 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %20 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %21 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %22 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %23 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %24 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %25 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %26 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %27 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x46")
    %28 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %29 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %30 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %31 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x45")
    %32 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x40")
    %33 = dfcir.shl(%32 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %34 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %35 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x38")
    %36 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x30")
    %37 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x27")
    %38 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %39 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %40 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x24")
    %41 = dfcir.shl(%40 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %42 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %43 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x20")
    %44 = dfcir.shl(%43 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %45 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %46 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %47 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x26")
    %48 = dfcir.mul[?] (%47 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %45 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %49 = dfcir.add[?] (%47 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %36 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %50 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x19")
    %51 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x51")
    %52 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %53 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %54 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x22")
    %55 = dfcir.mul[?] (%54 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %39 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %56 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x23")
    %57 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x2")
    %58 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x43")
    %59 = dfcir.add[?] (%31 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %58 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %60 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x7")
    %61 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %62 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 181 : si32
    %63 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1108 : si32
    %64 = dfcir.mul[?] (%49 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %65 = dfcir.add[?] (%64 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %48 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %66 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x48")
    %67 = dfcir.shl(%66 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %68 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %69 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %70 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x41")
    %71 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1609 : si32
    %72 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %73 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2408 : si32
    %74 = dfcir.mul[?] (%59 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %75 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x61")
    %76 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %77 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %78 = dfcir.mul[?] (%36 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %77 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %79 = dfcir.sub[?] (%64 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %78 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %80 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x39")
    %81 = dfcir.mul[?] (%80 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %38 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %82 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %83 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x33")
    %84 = dfcir.mul[?] (%83 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %85 = dfcir.add[?] (%83 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %80 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %86 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x25")
    %87 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %88 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x42")
    %89 = dfcir.add[?] (%88 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %27 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %90 = dfcir.mul[?] (%89 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %91 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x6")
    %92 = dfcir.add[?] (%57 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %91 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %93 = dfcir.mul[?] (%92 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %94 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %95 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x36")
    %96 = dfcir.shl(%95 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %97 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x12")
    %98 = dfcir.shl(%97 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %99 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x3")
    %100 = dfcir.mul[?] (%99 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %101 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %102 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 565 : si32
    %103 = dfcir.mul[?] (%85 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %104 = dfcir.sub[?] (%103 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %81 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %105 = dfcir.add[?] (%103 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %84 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %106 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x17")
    %107 = dfcir.add[?] (%106 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %56 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %108 = dfcir.mul[?] (%107 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %109 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %110 = dfcir.mul[?] (%106 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %111 = dfcir.add[?] (%108 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %110 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %112 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %113 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x47")
    %114 = dfcir.add[?] (%70 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %113 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %115 = dfcir.mul[?] (%114 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %116 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %117 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x4")
    %118 = dfcir.shl(%117 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %119 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %120 = dfcir.mul[?] (%27 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %119 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %121 = dfcir.sub[?] (%90 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %120 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %122 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x31")
    %123 = dfcir.mul[?] (%122 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %52 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %124 = dfcir.add[?] (%86 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %122 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %125 = dfcir.mul[?] (%124 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %126 = dfcir.sub[?] (%125 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %123 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %127 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x1")
    %128 = dfcir.mul[?] (%127 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %42 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %129 = dfcir.add[?] (%127 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %60 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %130 = dfcir.mul[?] (%129 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %131 = dfcir.add[?] (%130 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %128 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %132 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x44")
    %133 = dfcir.shl(%132 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %134 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %135 = dfcir.mul[?] (%58 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %134 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %136 = dfcir.sub[?] (%74 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %135 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %137 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4 : si32
    %138 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 8192 : si32
    %139 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x8")
    %140 = dfcir.shl(%139 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %141 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x34")
    %142 = dfcir.add[?] (%141 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %35 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %143 = dfcir.mul[?] (%142 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %144 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %145 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x18")
    %146 = dfcir.add[?] (%145 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %54 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %147 = dfcir.mul[?] (%146 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %148 = dfcir.sub[?] (%147 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %55 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %149 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x10")
    %150 = dfcir.mul[?] (%149 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %28 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %151 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x0")
    %152 = dfcir.shl(%151 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %153 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x35")
    %154 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %155 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x11")
    %156 = dfcir.mul[?] (%155 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %76 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %157 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x29")
    %158 = dfcir.add[?] (%157 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %37 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %159 = dfcir.mul[?] (%158 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %160 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x5")
    %161 = dfcir.mul[?] (%160 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %87 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %162 = dfcir.add[?] (%160 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %99 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %163 = dfcir.mul[?] (%162 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %164 = dfcir.sub[?] (%163 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %100 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %165 = dfcir.sub[?] (%163 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %161 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %166 = dfcir.sub[?] (%131 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %165 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %167 = dfcir.add[?] (%131 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %165 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %168 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x15")
    %169 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 128 : si32
    %170 = dfcir.add[?] (%67 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %171 = dfcir.add[?] (%33 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %172 = dfcir.sub[?] (%171 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %133 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %173 = dfcir.sub[?] (%172 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %121 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %174 = dfcir.add[?] (%172 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %121 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %175 = dfcir.add[?] (%171 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %133 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %176 = dfcir.add[?] (%41 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %177 = dfcir.add[?] (%140 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %178 = dfcir.sub[?] (%177 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %98 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %179 = dfcir.add[?] (%177 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %98 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %180 = dfcir.add[?] (%152 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %181 = dfcir.sub[?] (%180 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %118 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %182 = dfcir.add[?] (%180 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %118 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %183 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %184 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2841 : si32
    %185 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %186 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x16")
    %187 = dfcir.shl(%186 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %188 = dfcir.add[?] (%187 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %189 = dfcir.sub[?] (%188 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %44 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %190 = dfcir.sub[?] (%189 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %148 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %191 = dfcir.add[?] (%189 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %148 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %192 = dfcir.add[?] (%188 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %44 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %193 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x49")
    %194 = dfcir.mul[?] (%193 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %82 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %195 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x50")
    %196 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %197 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x52")
    %198 = dfcir.shl(%197 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %199 = dfcir.sub[?] (%170 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %198 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %200 = dfcir.add[?] (%170 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %198 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %201 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x53")
    %202 = dfcir.add[?] (%201 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %51 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %203 = dfcir.mul[?] (%202 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %204 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %205 = dfcir.mul[?] (%31 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %204 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %206 = dfcir.sub[?] (%74 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %205 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %207 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x54")
    %208 = dfcir.mul[?] (%207 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %30 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %209 = dfcir.add[?] (%195 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %207 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %210 = dfcir.mul[?] (%209 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %211 = dfcir.sub[?] (%210 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %212 = dfcir.sub[?] (%199 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %211 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %213 = dfcir.add[?] (%199 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %211 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %214 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x55")
    %215 = dfcir.add[?] (%193 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %214 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %216 = dfcir.mul[?] (%215 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %217 = dfcir.add[?] (%216 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %194 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %218 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x56")
    %219 = dfcir.shl(%218 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %220 = dfcir.add[?] (%219 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %221 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %222 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x57")
    %223 = dfcir.mul[?] (%222 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %183 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %224 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %225 = dfcir.mul[?] (%113 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %224 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %226 = dfcir.sub[?] (%115 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %225 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %227 = dfcir.sub[?] (%226 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %136 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %228 = dfcir.add[?] (%226 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %136 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %229 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x58")
    %230 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %231 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2676 : si32
    %232 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x59")
    %233 = dfcir.add[?] (%75 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %232 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %234 = dfcir.mul[?] (%233 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %235 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %236 = dfcir.mul[?] (%157 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %235 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %237 = dfcir.sub[?] (%159 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %236 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %238 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x62")
    %239 = dfcir.add[?] (%229 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %238 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %240 = dfcir.mul[?] (%239 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %241 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x37")
    %242 = dfcir.add[?] (%241 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %153 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %243 = dfcir.mul[?] (%242 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %244 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %245 = dfcir.mul[?] (%153 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %244 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %246 = dfcir.sub[?] (%243 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %245 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %247 = dfcir.sub[?] (%104 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %246 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %248 = dfcir.add[?] (%104 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %246 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %249 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x14")
    %250 = dfcir.mul[?] (%249 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %94 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %251 = dfcir.add[?] (%149 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %249 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %252 = dfcir.mul[?] (%251 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %253 = dfcir.add[?] (%252 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %150 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %254 = dfcir.sub[?] (%179 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %253 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %255 = dfcir.add[?] (%179 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %253 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %256 = dfcir.sub[?] (%252 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %250 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %257 = dfcir.sub[?] (%178 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %256 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %258 = dfcir.add[?] (%178 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %256 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %259 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x63")
    %260 = dfcir.add[?] (%222 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %259 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %261 = dfcir.mul[?] (%260 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %262 = dfcir.add[?] (%261 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %223 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %263 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x9")
    %264 = dfcir.mul[?] (%263 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %230 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %265 = dfcir.add[?] (%263 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %168 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %266 = dfcir.mul[?] (%265 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %267 = dfcir.add[?] (%266 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %264 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %268 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %269 = dfcir.mul[?] (%60 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %268 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %270 = dfcir.sub[?] (%130 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %269 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %271 = dfcir.sub[?] (%270 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %164 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %272 = dfcir.sub[?] (%166 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %271 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %273 = dfcir.mul[?] (%272 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %274 = dfcir.add[?] (%273 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %275 = dfcir.shr(%274 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %276 = dfcir.add[?] (%166 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %271 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %277 = dfcir.mul[?] (%276 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %278 = dfcir.add[?] (%277 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %279 = dfcir.shr(%278 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %280 = dfcir.add[?] (%270 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %164 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %281 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %282 = dfcir.mul[?] (%91 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %281 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %283 = dfcir.sub[?] (%93 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %282 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %284 = dfcir.sub[?] (%181 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %283 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %285 = dfcir.sub[?] (%284 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %275 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %286 = dfcir.shr(%285 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %287 = dfcir.shl(%286 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %288 = dfcir.add[?] (%287 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %289 = dfcir.add[?] (%284 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %275 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %290 = dfcir.shr(%289 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %291 = dfcir.shl(%290 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %292 = dfcir.add[?] (%291 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %293 = dfcir.add[?] (%181 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %283 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %294 = dfcir.sub[?] (%293 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %279 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %295 = dfcir.shr(%294 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %296 = dfcir.shl(%295 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %297 = dfcir.add[?] (%296 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %298 = dfcir.add[?] (%293 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %279 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %299 = dfcir.shr(%298 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %300 = dfcir.shl(%299 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %301 = dfcir.add[?] (%300 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %302 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %303 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %304 = dfcir.mul[?] (%57 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %303 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %305 = dfcir.add[?] (%93 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %304 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %306 = dfcir.sub[?] (%182 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %305 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %307 = dfcir.sub[?] (%306 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %280 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %308 = dfcir.shr(%307 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %309 = dfcir.shl(%308 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %310 = dfcir.add[?] (%309 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %311 = dfcir.add[?] (%306 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %280 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %312 = dfcir.shr(%311 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %313 = dfcir.shl(%312 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %314 = dfcir.add[?] (%313 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %315 = dfcir.add[?] (%182 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %305 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %316 = dfcir.sub[?] (%315 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %167 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %317 = dfcir.shr(%316 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %318 = dfcir.shl(%317 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %319 = dfcir.add[?] (%318 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %320 = dfcir.add[?] (%315 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %167 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %321 = dfcir.shr(%320 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %322 = dfcir.shl(%321 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %323 = dfcir.add[?] (%322 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %324 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %325 = dfcir.mul[?] (%37 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %324 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %326 = dfcir.sub[?] (%159 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %325 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %327 = dfcir.sub[?] (%126 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %326 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %328 = dfcir.add[?] (%126 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %326 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %329 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %330 = dfcir.mul[?] (%168 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %329 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %331 = dfcir.sub[?] (%266 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %330 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %332 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x13")
    %333 = dfcir.mul[?] (%332 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %154 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %334 = dfcir.add[?] (%332 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %155 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %335 = dfcir.mul[?] (%334 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %336 = dfcir.sub[?] (%335 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %156 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %337 = dfcir.sub[?] (%331 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %336 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %338 = dfcir.add[?] (%331 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %336 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %339 = dfcir.sub[?] (%254 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %338 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %340 = dfcir.shr(%339 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %341 = dfcir.mul[?] (%340 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %144 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %342 = dfcir.add[?] (%254 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %338 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %343 = dfcir.shr(%342 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %344 = dfcir.sub[?] (%335 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %333 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %345 = dfcir.sub[?] (%267 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %344 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %346 = dfcir.sub[?] (%345 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %337 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %347 = dfcir.mul[?] (%346 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %348 = dfcir.add[?] (%347 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %349 = dfcir.shr(%348 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %350 = dfcir.sub[?] (%257 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %349 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %351 = dfcir.shr(%350 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %352 = dfcir.mul[?] (%351 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %9 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %353 = dfcir.add[?] (%257 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %349 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %354 = dfcir.shr(%353 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %355 = dfcir.mul[?] (%354 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %19 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %356 = dfcir.add[?] (%345 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %337 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %357 = dfcir.mul[?] (%356 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %358 = dfcir.add[?] (%357 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %359 = dfcir.shr(%358 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %360 = dfcir.sub[?] (%258 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %359 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %361 = dfcir.shr(%360 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %362 = dfcir.mul[?] (%361 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %363 = dfcir.add[?] (%258 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %359 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %364 = dfcir.shr(%363 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %365 = dfcir.mul[?] (%364 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %25 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %366 = dfcir.add[?] (%267 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %344 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %367 = dfcir.sub[?] (%255 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %366 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %368 = dfcir.shr(%367 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %369 = dfcir.mul[?] (%368 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %370 = dfcir.add[?] (%255 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %366 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %371 = dfcir.shr(%370 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %372 = dfcir.mul[?] (%371 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %185 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %373 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %374 = dfcir.mul[?] (%56 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %373 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %375 = dfcir.sub[?] (%108 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %374 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %376 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %377 = dfcir.mul[?] (%50 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %376 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %378 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %379 = dfcir.mul[?] (%145 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %378 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %380 = dfcir.add[?] (%147 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %379 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %381 = dfcir.sub[?] (%192 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %380 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %382 = dfcir.add[?] (%192 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %380 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %383 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %384 = dfcir.mul[?] (%86 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %383 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %385 = dfcir.add[?] (%125 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %384 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %386 = dfcir.sub[?] (%385 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %237 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %387 = dfcir.sub[?] (%386 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %327 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %388 = dfcir.mul[?] (%387 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %389 = dfcir.add[?] (%388 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %390 = dfcir.shr(%389 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %391 = dfcir.add[?] (%386 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %327 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %392 = dfcir.mul[?] (%391 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %393 = dfcir.add[?] (%392 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %394 = dfcir.shr(%393 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %395 = dfcir.add[?] (%385 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %237 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %396 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %397 = dfcir.mul[?] (%229 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %396 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %398 = dfcir.add[?] (%240 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %397 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %399 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %400 = dfcir.mul[?] (%35 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %399 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %401 = dfcir.sub[?] (%143 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %400 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %402 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %403 = dfcir.mul[?] (%343 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %402 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %404 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x21")
    %405 = dfcir.mul[?] (%404 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %72 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %406 = dfcir.add[?] (%404 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %50 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %407 = dfcir.mul[?] (%406 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %408 = dfcir.sub[?] (%407 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %377 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %409 = dfcir.sub[?] (%375 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %408 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %410 = dfcir.add[?] (%375 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %408 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %411 = dfcir.sub[?] (%381 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %410 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %412 = dfcir.shr(%411 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %413 = dfcir.mul[?] (%412 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %46 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %414 = dfcir.add[?] (%381 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %410 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %415 = dfcir.shr(%414 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %416 = dfcir.mul[?] (%415 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %21 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %417 = dfcir.sub[?] (%407 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %405 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %418 = dfcir.sub[?] (%111 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %417 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %419 = dfcir.sub[?] (%418 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %409 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %420 = dfcir.mul[?] (%419 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %421 = dfcir.add[?] (%420 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %422 = dfcir.shr(%421 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %423 = dfcir.sub[?] (%190 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %422 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %424 = dfcir.shr(%423 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %425 = dfcir.mul[?] (%424 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %426 = dfcir.add[?] (%190 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %422 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %427 = dfcir.shr(%426 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %428 = dfcir.mul[?] (%427 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %101 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %429 = dfcir.add[?] (%418 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %409 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %430 = dfcir.mul[?] (%429 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %431 = dfcir.add[?] (%430 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %432 = dfcir.shr(%431 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %433 = dfcir.sub[?] (%191 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %432 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %434 = dfcir.shr(%433 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %435 = dfcir.mul[?] (%434 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %221 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %436 = dfcir.add[?] (%191 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %432 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %437 = dfcir.shr(%436 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %438 = dfcir.mul[?] (%437 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %34 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %439 = dfcir.add[?] (%111 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %417 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %440 = dfcir.sub[?] (%382 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %439 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %441 = dfcir.shr(%440 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %442 = dfcir.add[?] (%382 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %439 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %443 = dfcir.shr(%442 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %444 = dfcir.mul[?] (%443 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %26 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %445 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2276 : si32
    %446 = dfcir.mul[?] (%70 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %445 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %447 = dfcir.add[?] (%115 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %446 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %448 = dfcir.sub[?] (%447 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %206 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %449 = dfcir.sub[?] (%448 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %227 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %450 = dfcir.mul[?] (%449 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %451 = dfcir.add[?] (%450 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %452 = dfcir.shr(%451 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %453 = dfcir.sub[?] (%173 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %452 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %454 = dfcir.shr(%453 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %455 = dfcir.mul[?] (%454 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %112 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %456 = dfcir.add[?] (%173 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %452 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %457 = dfcir.shr(%456 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %458 = dfcir.mul[?] (%457 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %18 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %459 = dfcir.add[?] (%448 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %227 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %460 = dfcir.mul[?] (%459 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %461 = dfcir.add[?] (%460 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %462 = dfcir.shr(%461 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %463 = dfcir.sub[?] (%174 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %462 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %464 = dfcir.shr(%463 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %465 = dfcir.mul[?] (%464 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %29 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %466 = dfcir.add[?] (%174 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %462 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %467 = dfcir.shr(%466 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %468 = dfcir.mul[?] (%467 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %24 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %469 = dfcir.add[?] (%447 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %206 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %470 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %471 = dfcir.mul[?] (%141 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %470 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %472 = dfcir.add[?] (%143 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %471 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %473 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %474 = dfcir.mul[?] (%88 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %473 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %475 = dfcir.add[?] (%90 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %474 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %476 = dfcir.sub[?] (%175 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %475 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %477 = dfcir.sub[?] (%476 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %228 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %478 = dfcir.shr(%477 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %479 = dfcir.mul[?] (%478 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %480 = dfcir.add[?] (%476 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %228 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %481 = dfcir.shr(%480 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %482 = dfcir.mul[?] (%481 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %483 = dfcir.add[?] (%175 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %475 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %484 = dfcir.sub[?] (%483 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %469 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %485 = dfcir.shr(%484 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %486 = dfcir.mul[?] (%485 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %487 = dfcir.add[?] (%483 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %469 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %488 = dfcir.shr(%487 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %489 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %490 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %491 = dfcir.mul[?] (%214 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %490 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %492 = dfcir.sub[?] (%216 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %491 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %493 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x28")
    %494 = dfcir.shl(%493 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %495 = dfcir.sub[?] (%176 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %494 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %496 = dfcir.sub[?] (%495 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %79 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %497 = dfcir.sub[?] (%496 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %390 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %498 = dfcir.shr(%497 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %499 = dfcir.mul[?] (%498 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %489 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %500 = dfcir.add[?] (%454 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %498 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %501 = dfcir.mul[?] (%500 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %502 = dfcir.add[?] (%501 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %503 = dfcir.sub[?] (%502 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %499 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %504 = dfcir.shr(%503 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %505 = dfcir.sub[?] (%502 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %455 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %506 = dfcir.shr(%505 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %507 = dfcir.add[?] (%496 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %390 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %508 = dfcir.shr(%507 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %509 = dfcir.mul[?] (%508 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %17 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %510 = dfcir.add[?] (%457 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %508 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %511 = dfcir.mul[?] (%510 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %512 = dfcir.add[?] (%511 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %513 = dfcir.sub[?] (%512 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %509 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %514 = dfcir.shr(%513 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %515 = dfcir.sub[?] (%512 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %458 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %516 = dfcir.shr(%515 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %517 = dfcir.add[?] (%495 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %79 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %518 = dfcir.sub[?] (%517 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %394 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %519 = dfcir.shr(%518 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %520 = dfcir.mul[?] (%519 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %521 = dfcir.add[?] (%464 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %519 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %522 = dfcir.mul[?] (%521 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %523 = dfcir.add[?] (%522 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %524 = dfcir.sub[?] (%523 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %520 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %525 = dfcir.shr(%524 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %526 = dfcir.sub[?] (%523 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %465 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %527 = dfcir.shr(%526 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %528 = dfcir.add[?] (%517 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %394 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %529 = dfcir.shr(%528 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %530 = dfcir.mul[?] (%529 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %22 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %531 = dfcir.add[?] (%467 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %529 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %532 = dfcir.mul[?] (%531 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %533 = dfcir.add[?] (%532 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %534 = dfcir.sub[?] (%533 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %530 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %535 = dfcir.shr(%534 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %536 = dfcir.sub[?] (%533 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %468 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %537 = dfcir.shr(%536 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %538 = dfcir.add[?] (%176 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %494 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %539 = dfcir.sub[?] (%538 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %65 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %540 = dfcir.sub[?] (%539 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %328 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %541 = dfcir.shr(%540 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %542 = dfcir.mul[?] (%541 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %61 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %543 = dfcir.add[?] (%478 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %541 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %544 = dfcir.mul[?] (%543 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %545 = dfcir.add[?] (%544 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %546 = dfcir.sub[?] (%545 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %542 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %547 = dfcir.shr(%546 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %548 = dfcir.sub[?] (%545 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %479 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %549 = dfcir.shr(%548 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %550 = dfcir.add[?] (%539 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %328 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %551 = dfcir.shr(%550 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %552 = dfcir.mul[?] (%551 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %553 = dfcir.add[?] (%481 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %551 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %554 = dfcir.mul[?] (%553 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %555 = dfcir.add[?] (%554 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %556 = dfcir.sub[?] (%555 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %552 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %557 = dfcir.shr(%556 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %558 = dfcir.sub[?] (%555 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %482 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %559 = dfcir.shr(%558 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %560 = dfcir.add[?] (%538 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %65 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %561 = dfcir.sub[?] (%560 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %395 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %562 = dfcir.shr(%561 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %563 = dfcir.mul[?] (%562 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %564 = dfcir.add[?] (%485 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %562 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %565 = dfcir.mul[?] (%564 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %566 = dfcir.add[?] (%565 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %567 = dfcir.sub[?] (%566 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %563 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %568 = dfcir.shr(%567 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %569 = dfcir.sub[?] (%566 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %486 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %570 = dfcir.shr(%569 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %571 = dfcir.add[?] (%560 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %395 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %572 = dfcir.shr(%571 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %573 = dfcir.mul[?] (%572 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %196 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %574 = dfcir.add[?] (%488 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %572 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %575 = dfcir.mul[?] (%574 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %576 = dfcir.add[?] (%575 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %577 = dfcir.sub[?] (%576 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %573 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %578 = dfcir.shr(%577 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %579 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %580 = dfcir.mul[?] (%201 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %579 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %581 = dfcir.sub[?] (%203 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %580 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %582 = dfcir.sub[?] (%217 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %581 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %583 = dfcir.add[?] (%217 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %581 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %584 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %585 = dfcir.mul[?] (%51 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %584 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %586 = dfcir.sub[?] (%203 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %585 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %587 = dfcir.sub[?] (%492 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %586 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %588 = dfcir.sub[?] (%582 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %587 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %589 = dfcir.mul[?] (%588 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %590 = dfcir.add[?] (%589 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %591 = dfcir.shr(%590 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %592 = dfcir.sub[?] (%212 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %591 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %593 = dfcir.shr(%592 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %594 = dfcir.mul[?] (%593 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %595 = dfcir.add[?] (%424 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %593 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %596 = dfcir.mul[?] (%595 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %597 = dfcir.add[?] (%596 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %598 = dfcir.add[?] (%597 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %425 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %599 = dfcir.shr(%598 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %600 = dfcir.sub[?] (%597 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %594 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %601 = dfcir.shr(%600 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %602 = dfcir.add[?] (%212 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %591 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %603 = dfcir.shr(%602 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %604 = dfcir.mul[?] (%603 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %16 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %605 = dfcir.add[?] (%427 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %603 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %606 = dfcir.mul[?] (%605 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %607 = dfcir.add[?] (%606 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %608 = dfcir.add[?] (%607 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %428 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %609 = dfcir.shr(%608 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %610 = dfcir.sub[?] (%607 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %604 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %611 = dfcir.shr(%610 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %612 = dfcir.add[?] (%582 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %587 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %613 = dfcir.mul[?] (%612 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %614 = dfcir.add[?] (%613 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %615 = dfcir.shr(%614 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %616 = dfcir.sub[?] (%213 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %615 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %617 = dfcir.shr(%616 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %618 = dfcir.mul[?] (%617 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %619 = dfcir.add[?] (%434 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %617 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %620 = dfcir.mul[?] (%619 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %621 = dfcir.add[?] (%620 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %622 = dfcir.add[?] (%621 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %435 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %623 = dfcir.shr(%622 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %624 = dfcir.sub[?] (%621 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %618 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %625 = dfcir.shr(%624 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %626 = dfcir.add[?] (%213 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %615 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %627 = dfcir.shr(%626 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %628 = dfcir.mul[?] (%627 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %20 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %629 = dfcir.add[?] (%437 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %627 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %630 = dfcir.mul[?] (%629 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %631 = dfcir.add[?] (%630 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %632 = dfcir.add[?] (%631 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %438 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %633 = dfcir.shr(%632 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %634 = dfcir.sub[?] (%631 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %628 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %635 = dfcir.shr(%634 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %636 = dfcir.add[?] (%492 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %586 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %637 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %638 = dfcir.mul[?] (%195 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %637 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %639 = dfcir.add[?] (%210 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %638 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %640 = dfcir.sub[?] (%200 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %639 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %641 = dfcir.sub[?] (%640 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %636 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %642 = dfcir.shr(%641 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %643 = dfcir.mul[?] (%642 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %644 = dfcir.add[?] (%412 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %642 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %645 = dfcir.mul[?] (%644 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %646 = dfcir.add[?] (%645 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %647 = dfcir.add[?] (%646 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %413 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %648 = dfcir.shr(%647 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %649 = dfcir.sub[?] (%646 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %643 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %650 = dfcir.shr(%649 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %651 = dfcir.add[?] (%640 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %636 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %652 = dfcir.shr(%651 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %653 = dfcir.mul[?] (%652 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %23 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %654 = dfcir.add[?] (%415 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %652 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %655 = dfcir.mul[?] (%654 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %656 = dfcir.add[?] (%655 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %657 = dfcir.add[?] (%656 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %416 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %658 = dfcir.shr(%657 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %659 = dfcir.sub[?] (%656 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %653 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %660 = dfcir.shr(%659 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %661 = dfcir.add[?] (%200 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %639 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %662 = dfcir.sub[?] (%661 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %583 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %663 = dfcir.shr(%662 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %664 = dfcir.mul[?] (%663 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %665 = dfcir.add[?] (%441 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %663 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %666 = dfcir.mul[?] (%665 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %667 = dfcir.add[?] (%666 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %668 = dfcir.sub[?] (%667 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %664 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %669 = dfcir.shr(%668 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %670 = dfcir.add[?] (%661 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %583 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %671 = dfcir.shr(%670 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %672 = dfcir.add[?] (%443 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %671 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %673 = dfcir.mul[?] (%672 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %674 = dfcir.add[?] (%673 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %675 = dfcir.add[?] (%674 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %444 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %676 = dfcir.shr(%675 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %677 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %678 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %679 = dfcir.mul[?] (%259 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %678 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %680 = dfcir.sub[?] (%261 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %679 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %681 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %682 = dfcir.mul[?] (%75 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %681 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %683 = dfcir.sub[?] (%234 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %682 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %684 = dfcir.sub[?] (%262 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %683 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %685 = dfcir.add[?] (%262 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %683 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %686 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %687 = dfcir.mul[?] (%241 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %686 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %688 = dfcir.sub[?] (%243 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %687 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %689 = dfcir.sub[?] (%105 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %688 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %690 = dfcir.sub[?] (%689 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %247 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %691 = dfcir.mul[?] (%690 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %692 = dfcir.add[?] (%691 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %693 = dfcir.shr(%692 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %694 = dfcir.add[?] (%689 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %247 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %695 = dfcir.mul[?] (%694 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %696 = dfcir.add[?] (%695 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %697 = dfcir.shr(%696 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %698 = dfcir.add[?] (%105 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %688 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %699 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4017 : si32
    %700 = dfcir.mul[?] (%232 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %699 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %701 = dfcir.sub[?] (%234 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %700 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %702 = dfcir.sub[?] (%680 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %701 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %703 = dfcir.sub[?] (%684 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %702 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %704 = dfcir.mul[?] (%703 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %705 = dfcir.add[?] (%704 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %706 = dfcir.shr(%705 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %707 = dfcir.add[?] (%684 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %702 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %708 = dfcir.mul[?] (%707 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %709 = dfcir.add[?] (%708 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %710 = dfcir.shr(%709 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %711 = dfcir.add[?] (%680 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %701 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %712 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %713 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x32")
    %714 = dfcir.shl(%713 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %715 = dfcir.add[?] (%714 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %716 = dfcir.sub[?] (%715 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %96 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %717 = dfcir.sub[?] (%716 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %401 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %718 = dfcir.sub[?] (%717 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %693 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %719 = dfcir.shr(%718 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %720 = dfcir.shl(%719 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %721 = dfcir.sub[?] (%288 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %720 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %722 = dfcir.sub[?] (%721 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %601 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %723 = dfcir.add[?] (%721 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %601 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %724 = dfcir.add[?] (%288 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %720 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %725 = dfcir.sub[?] (%724 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %599 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %726 = dfcir.add[?] (%724 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %599 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %727 = dfcir.add[?] (%717 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %693 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %728 = dfcir.shr(%727 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %729 = dfcir.shl(%728 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %730 = dfcir.sub[?] (%292 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %729 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %731 = dfcir.sub[?] (%730 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %611 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %732 = dfcir.add[?] (%730 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %611 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %733 = dfcir.add[?] (%292 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %729 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %734 = dfcir.sub[?] (%733 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %609 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %735 = dfcir.add[?] (%733 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %609 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %736 = dfcir.add[?] (%716 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %401 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %737 = dfcir.sub[?] (%736 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %697 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %738 = dfcir.shr(%737 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %739 = dfcir.shl(%738 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %740 = dfcir.sub[?] (%297 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %739 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %741 = dfcir.sub[?] (%740 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %625 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %742 = dfcir.add[?] (%740 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %625 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %743 = dfcir.add[?] (%297 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %739 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %744 = dfcir.sub[?] (%743 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %623 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %745 = dfcir.add[?] (%743 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %623 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %746 = dfcir.add[?] (%736 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %697 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %747 = dfcir.shr(%746 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %748 = dfcir.shl(%747 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %749 = dfcir.sub[?] (%301 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %748 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %750 = dfcir.sub[?] (%749 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %635 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %751 = dfcir.add[?] (%749 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %635 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %752 = dfcir.add[?] (%301 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %748 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %753 = dfcir.sub[?] (%752 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %633 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %754 = dfcir.add[?] (%752 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %633 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %755 = dfcir.add[?] (%715 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %96 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %756 = dfcir.sub[?] (%755 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %472 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %757 = dfcir.sub[?] (%756 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %248 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %758 = dfcir.shr(%757 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %759 = dfcir.shl(%758 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %760 = dfcir.sub[?] (%310 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %759 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %761 = dfcir.sub[?] (%760 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %650 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %762 = dfcir.add[?] (%760 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %650 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %763 = dfcir.add[?] (%310 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %759 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %764 = dfcir.sub[?] (%763 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %648 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %765 = dfcir.add[?] (%763 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %648 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %766 = dfcir.add[?] (%756 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %248 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %767 = dfcir.shr(%766 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %768 = dfcir.shl(%767 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %769 = dfcir.sub[?] (%314 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %768 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %770 = dfcir.sub[?] (%769 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %660 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %771 = dfcir.add[?] (%769 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %660 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %772 = dfcir.add[?] (%314 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %768 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %773 = dfcir.sub[?] (%772 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %658 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %774 = dfcir.add[?] (%772 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %658 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %775 = dfcir.add[?] (%755 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %472 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %776 = dfcir.sub[?] (%775 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %698 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %777 = dfcir.shr(%776 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %778 = dfcir.shl(%777 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %779 = dfcir.sub[?] (%319 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %778 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %780 = dfcir.sub[?] (%779 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %669 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %781 = dfcir.add[?] (%779 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %669 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %782 = dfcir.add[?] (%319 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %778 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %783 = dfcir.add[?] (%775 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %698 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %784 = dfcir.shr(%783 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %785 = dfcir.shl(%784 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %786 = dfcir.sub[?] (%323 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %785 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %787 = dfcir.add[?] (%323 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %785 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %788 = dfcir.sub[?] (%787 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %676 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %789 = dfcir.add[?] (%787 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %676 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %790 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %791 = dfcir.mul[?] (%238 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %790 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %792 = dfcir.sub[?] (%240 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %791 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %793 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3406 : si32
    %794 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x60")
    %795 = dfcir.shl(%794 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %796 = dfcir.sub[?] (%220 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %795 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %797 = dfcir.sub[?] (%796 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %792 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %798 = dfcir.sub[?] (%797 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %706 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %799 = dfcir.shr(%798 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %800 = dfcir.mul[?] (%799 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %677 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %801 = dfcir.add[?] (%351 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %799 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %802 = dfcir.mul[?] (%801 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %803 = dfcir.add[?] (%802 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %804 = dfcir.sub[?] (%803 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %800 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %805 = dfcir.shr(%804 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %806 = dfcir.sub[?] (%805 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %504 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %807 = dfcir.add[?] (%805 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %504 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %808 = dfcir.sub[?] (%725 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %807 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %809 = dfcir.shr(%808 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %810 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out37") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%810 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %809 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %811 = dfcir.add[?] (%725 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %807 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %812 = dfcir.shr(%811 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %813 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out29") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%813 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %812 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %814 = dfcir.add[?] (%803 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %352 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %815 = dfcir.shr(%814 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %816 = dfcir.sub[?] (%815 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %506 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %817 = dfcir.sub[?] (%816 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %806 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %818 = dfcir.mul[?] (%817 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %819 = dfcir.add[?] (%818 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %820 = dfcir.shr(%819 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %821 = dfcir.sub[?] (%722 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %820 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %822 = dfcir.shr(%821 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %823 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out45") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%823 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %822 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %824 = dfcir.add[?] (%722 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %820 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %825 = dfcir.shr(%824 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %826 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out21") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%826 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %825 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %827 = dfcir.add[?] (%816 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %806 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %828 = dfcir.mul[?] (%827 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %829 = dfcir.add[?] (%828 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %830 = dfcir.shr(%829 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %831 = dfcir.sub[?] (%723 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %830 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %832 = dfcir.shr(%831 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %833 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out53") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%833 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %832 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %834 = dfcir.add[?] (%723 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %830 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %835 = dfcir.shr(%834 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %836 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out13") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%836 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %835 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %837 = dfcir.add[?] (%815 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %506 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %838 = dfcir.sub[?] (%726 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %837 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %839 = dfcir.shr(%838 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %840 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out61") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%840 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %839 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %841 = dfcir.add[?] (%726 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %837 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %842 = dfcir.shr(%841 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %843 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out5") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%843 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %842 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %844 = dfcir.add[?] (%797 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %706 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %845 = dfcir.shr(%844 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %846 = dfcir.mul[?] (%845 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %712 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %847 = dfcir.add[?] (%354 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %845 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %848 = dfcir.mul[?] (%847 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %849 = dfcir.add[?] (%848 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %850 = dfcir.sub[?] (%849 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %846 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %851 = dfcir.shr(%850 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %852 = dfcir.sub[?] (%851 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %514 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %853 = dfcir.add[?] (%851 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %514 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %854 = dfcir.sub[?] (%734 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %853 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %855 = dfcir.shr(%854 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %856 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out34") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%856 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %855 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %857 = dfcir.add[?] (%734 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %853 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %858 = dfcir.shr(%857 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %859 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out26") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%859 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %858 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %860 = dfcir.add[?] (%849 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %355 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %861 = dfcir.shr(%860 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %862 = dfcir.sub[?] (%861 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %516 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %863 = dfcir.sub[?] (%862 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %852 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %864 = dfcir.mul[?] (%863 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %865 = dfcir.add[?] (%864 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %866 = dfcir.shr(%865 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %867 = dfcir.sub[?] (%731 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %866 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %868 = dfcir.shr(%867 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %869 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out42") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%869 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %868 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %870 = dfcir.add[?] (%731 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %866 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %871 = dfcir.shr(%870 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %872 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out18") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%872 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %871 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %873 = dfcir.add[?] (%862 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %852 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %874 = dfcir.mul[?] (%873 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %875 = dfcir.add[?] (%874 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %876 = dfcir.shr(%875 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %877 = dfcir.sub[?] (%732 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %876 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %878 = dfcir.shr(%877 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %879 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out50") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%879 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %878 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %880 = dfcir.add[?] (%732 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %876 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %881 = dfcir.shr(%880 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %882 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out10") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%882 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %881 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %883 = dfcir.add[?] (%861 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %516 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %884 = dfcir.sub[?] (%735 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %883 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %885 = dfcir.shr(%884 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %886 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out58") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%886 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %885 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %887 = dfcir.add[?] (%735 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %883 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %888 = dfcir.shr(%887 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %889 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out2") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%889 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %888 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %890 = dfcir.add[?] (%796 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %792 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %891 = dfcir.sub[?] (%890 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %710 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %892 = dfcir.shr(%891 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %893 = dfcir.mul[?] (%892 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %69 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %894 = dfcir.add[?] (%361 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %892 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %895 = dfcir.mul[?] (%894 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %896 = dfcir.add[?] (%895 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %897 = dfcir.sub[?] (%896 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %893 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %898 = dfcir.shr(%897 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %899 = dfcir.sub[?] (%898 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %525 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %900 = dfcir.add[?] (%898 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %525 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %901 = dfcir.sub[?] (%744 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %900 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %902 = dfcir.shr(%901 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %903 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out38") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%903 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %902 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %904 = dfcir.add[?] (%744 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %900 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %905 = dfcir.shr(%904 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %906 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out30") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%906 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %905 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %907 = dfcir.add[?] (%896 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %362 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %908 = dfcir.shr(%907 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %909 = dfcir.sub[?] (%908 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %527 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %910 = dfcir.sub[?] (%909 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %899 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %911 = dfcir.mul[?] (%910 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %912 = dfcir.add[?] (%911 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %913 = dfcir.shr(%912 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %914 = dfcir.sub[?] (%741 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %913 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %915 = dfcir.shr(%914 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %916 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out46") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%916 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %915 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %917 = dfcir.add[?] (%741 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %913 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %918 = dfcir.shr(%917 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %919 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out22") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%919 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %918 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %920 = dfcir.add[?] (%909 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %899 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %921 = dfcir.mul[?] (%920 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %922 = dfcir.add[?] (%921 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %923 = dfcir.shr(%922 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %924 = dfcir.sub[?] (%742 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %923 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %925 = dfcir.shr(%924 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %926 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out54") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%926 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %925 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %927 = dfcir.add[?] (%742 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %923 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %928 = dfcir.shr(%927 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %929 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out14") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%929 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %928 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %930 = dfcir.add[?] (%908 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %527 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %931 = dfcir.sub[?] (%745 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %930 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %932 = dfcir.shr(%931 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %933 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out62") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%933 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %932 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %934 = dfcir.add[?] (%745 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %930 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %935 = dfcir.shr(%934 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %936 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out6") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%936 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %935 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %937 = dfcir.add[?] (%890 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %710 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %938 = dfcir.shr(%937 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %939 = dfcir.mul[?] (%938 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %302 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %940 = dfcir.add[?] (%364 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %938 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %941 = dfcir.mul[?] (%940 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %942 = dfcir.add[?] (%941 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %943 = dfcir.sub[?] (%942 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %939 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %944 = dfcir.shr(%943 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %945 = dfcir.sub[?] (%944 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %535 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %946 = dfcir.add[?] (%944 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %535 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %947 = dfcir.sub[?] (%753 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %946 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %948 = dfcir.shr(%947 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %949 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out33") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%949 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %948 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %950 = dfcir.add[?] (%753 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %946 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %951 = dfcir.shr(%950 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %952 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out25") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%952 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %951 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %953 = dfcir.add[?] (%942 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %365 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %954 = dfcir.shr(%953 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %955 = dfcir.sub[?] (%954 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %537 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %956 = dfcir.sub[?] (%955 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %945 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %957 = dfcir.mul[?] (%956 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %958 = dfcir.add[?] (%957 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %959 = dfcir.shr(%958 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %960 = dfcir.sub[?] (%750 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %959 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %961 = dfcir.shr(%960 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %962 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out41") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%962 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %961 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %963 = dfcir.add[?] (%750 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %959 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %964 = dfcir.shr(%963 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %965 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out17") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%965 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %964 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %966 = dfcir.add[?] (%955 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %945 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %967 = dfcir.mul[?] (%966 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %968 = dfcir.add[?] (%967 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %969 = dfcir.shr(%968 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %970 = dfcir.sub[?] (%751 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %969 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %971 = dfcir.shr(%970 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %972 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out49") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%972 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %971 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %973 = dfcir.add[?] (%751 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %969 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %974 = dfcir.shr(%973 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %975 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out9") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%975 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %974 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %976 = dfcir.add[?] (%954 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %537 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %977 = dfcir.sub[?] (%754 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %976 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %978 = dfcir.shr(%977 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %979 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out57") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%979 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %978 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %980 = dfcir.add[?] (%754 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %976 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %981 = dfcir.shr(%980 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %982 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out1") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%982 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %981 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %983 = dfcir.add[?] (%220 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %795 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %984 = dfcir.sub[?] (%983 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %398 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %985 = dfcir.sub[?] (%984 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %711 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %986 = dfcir.shr(%985 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %987 = dfcir.mul[?] (%986 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %988 = dfcir.add[?] (%340 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %986 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %989 = dfcir.mul[?] (%988 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %990 = dfcir.add[?] (%989 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %991 = dfcir.sub[?] (%990 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %987 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %992 = dfcir.shr(%991 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %993 = dfcir.sub[?] (%992 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %547 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %994 = dfcir.add[?] (%992 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %547 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %995 = dfcir.sub[?] (%764 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %994 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %996 = dfcir.shr(%995 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %997 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out36") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%997 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %996 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %998 = dfcir.add[?] (%764 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %994 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %999 = dfcir.shr(%998 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1000 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out28") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1000 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %999 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1001 = dfcir.add[?] (%990 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %341 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1002 = dfcir.shr(%1001 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1003 = dfcir.sub[?] (%1002 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %549 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1004 = dfcir.sub[?] (%1003 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %993 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1005 = dfcir.mul[?] (%1004 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1006 = dfcir.add[?] (%1005 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1007 = dfcir.shr(%1006 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1008 = dfcir.sub[?] (%761 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1007 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1009 = dfcir.shr(%1008 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1010 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out44") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1010 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1009 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1011 = dfcir.add[?] (%761 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1007 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1012 = dfcir.shr(%1011 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1013 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out20") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1013 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1012 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1014 = dfcir.add[?] (%1003 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %993 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1015 = dfcir.mul[?] (%1014 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1016 = dfcir.add[?] (%1015 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1017 = dfcir.shr(%1016 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1018 = dfcir.sub[?] (%762 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1017 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1019 = dfcir.shr(%1018 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1020 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out52") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1020 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1019 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1021 = dfcir.add[?] (%762 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1017 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1022 = dfcir.shr(%1021 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1023 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out12") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1023 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1022 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1024 = dfcir.add[?] (%1002 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %549 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1025 = dfcir.sub[?] (%765 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1024 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1026 = dfcir.shr(%1025 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1027 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out60") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1027 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1026 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1028 = dfcir.add[?] (%765 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1024 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1029 = dfcir.shr(%1028 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1030 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out4") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1030 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1029 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1031 = dfcir.add[?] (%984 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %711 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1032 = dfcir.shr(%1031 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1033 = dfcir.mul[?] (%1032 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %15 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1034 = dfcir.add[?] (%343 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1032 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1035 = dfcir.mul[?] (%1034 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1036 = dfcir.add[?] (%1035 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1037 = dfcir.sub[?] (%1036 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1033 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1038 = dfcir.shr(%1037 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1039 = dfcir.sub[?] (%1038 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %557 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1040 = dfcir.add[?] (%1038 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %557 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1041 = dfcir.sub[?] (%773 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1040 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1042 = dfcir.shr(%1041 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1043 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out35") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1043 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1042 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1044 = dfcir.add[?] (%773 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1040 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1045 = dfcir.shr(%1044 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1046 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out27") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1046 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1045 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1047 = dfcir.add[?] (%1036 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %403 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1048 = dfcir.shr(%1047 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1049 = dfcir.sub[?] (%1048 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %559 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1050 = dfcir.sub[?] (%1049 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1039 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1051 = dfcir.mul[?] (%1050 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1052 = dfcir.add[?] (%1051 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1053 = dfcir.shr(%1052 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1054 = dfcir.sub[?] (%770 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1053 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1055 = dfcir.shr(%1054 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1056 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out43") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1056 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1055 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1057 = dfcir.add[?] (%770 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1053 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1058 = dfcir.shr(%1057 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1059 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out19") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1059 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1058 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1060 = dfcir.add[?] (%1049 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1039 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1061 = dfcir.mul[?] (%1060 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1062 = dfcir.add[?] (%1061 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1063 = dfcir.shr(%1062 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1064 = dfcir.sub[?] (%771 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1063 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1065 = dfcir.shr(%1064 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1066 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out51") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1066 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1065 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1067 = dfcir.add[?] (%771 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1063 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1068 = dfcir.shr(%1067 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1069 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out11") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1069 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1068 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1070 = dfcir.add[?] (%1048 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %559 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1071 = dfcir.sub[?] (%774 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1070 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1072 = dfcir.shr(%1071 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1073 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out59") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1073 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1072 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1074 = dfcir.add[?] (%774 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1070 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1075 = dfcir.shr(%1074 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1076 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out3") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1076 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1075 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1077 = dfcir.add[?] (%983 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %398 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1078 = dfcir.sub[?] (%1077 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %685 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1079 = dfcir.shr(%1078 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1080 = dfcir.mul[?] (%1079 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1081 = dfcir.add[?] (%368 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1079 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1082 = dfcir.mul[?] (%1081 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1083 = dfcir.add[?] (%1082 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1084 = dfcir.sub[?] (%1083 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1080 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1085 = dfcir.shr(%1084 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1086 = dfcir.sub[?] (%1085 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %568 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1087 = dfcir.add[?] (%1085 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %568 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1088 = dfcir.add[?] (%1083 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %369 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1089 = dfcir.shr(%1088 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1090 = dfcir.sub[?] (%1089 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %570 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1091 = dfcir.sub[?] (%1090 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1086 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1092 = dfcir.mul[?] (%1091 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1093 = dfcir.add[?] (%1092 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1094 = dfcir.shr(%1093 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1095 = dfcir.sub[?] (%780 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1094 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1096 = dfcir.shr(%1095 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1097 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out47") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1097 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1096 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1098 = dfcir.add[?] (%780 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1094 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1099 = dfcir.shr(%1098 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1100 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out23") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1100 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1099 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1101 = dfcir.add[?] (%1090 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1086 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1102 = dfcir.mul[?] (%1101 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1103 = dfcir.add[?] (%1102 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1104 = dfcir.shr(%1103 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1105 = dfcir.sub[?] (%781 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1104 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1106 = dfcir.shr(%1105 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1107 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out55") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1107 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1106 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1108 = dfcir.add[?] (%781 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1104 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1109 = dfcir.shr(%1108 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1110 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out15") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1110 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1109 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1111 = dfcir.add[?] (%1089 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %570 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1112 = dfcir.add[?] (%1077 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %685 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1113 = dfcir.shr(%1112 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1114 = dfcir.mul[?] (%1113 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %793 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1115 = dfcir.add[?] (%371 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1113 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1116 = dfcir.mul[?] (%1115 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1117 = dfcir.add[?] (%1116 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %137 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1118 = dfcir.sub[?] (%1117 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1114 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1119 = dfcir.shr(%1118 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1120 = dfcir.sub[?] (%1119 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %578 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1121 = dfcir.add[?] (%1119 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %578 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1122 = dfcir.sub[?] (%788 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1121 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1123 = dfcir.shr(%1122 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1124 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out32") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1124 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1123 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1125 = dfcir.add[?] (%788 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1121 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1126 = dfcir.shr(%1125 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1127 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out24") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1127 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1126 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1128 = dfcir.add[?] (%1117 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %372 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1129 = dfcir.shr(%1128 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1130 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 799 : si32
    %1131 = dfcir.mul[?] (%488 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1130 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1132 = dfcir.sub[?] (%576 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1131 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1133 = dfcir.shr(%1132 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1134 = dfcir.sub[?] (%1129 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1133 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1135 = dfcir.sub[?] (%1134 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1120 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1136 = dfcir.mul[?] (%1135 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1137 = dfcir.add[?] (%1136 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1138 = dfcir.shr(%1137 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1139 = dfcir.add[?] (%1134 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1120 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1140 = dfcir.mul[?] (%1139 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1141 = dfcir.add[?] (%1140 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %169 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1142 = dfcir.shr(%1141 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1143 = dfcir.add[?] (%1129 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1133 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>> {latency = -1 : i32}
    %1144 = dfcir.sub[?] (%789 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1143 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1145 = dfcir.shr(%1144 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1146 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out56") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1146 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1145 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1147 = dfcir.add[?] (%789 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1143 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1148 = dfcir.shr(%1147 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1149 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out0") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1149 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1148 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1150 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1568 : si32
    %1151 = dfcir.mul[?] (%441 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1152 = dfcir.add[?] (%667 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1151 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1153 = dfcir.shr(%1152 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1154 = dfcir.sub[?] (%782 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1153 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1155 = dfcir.sub[?] (%1154 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1087 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1156 = dfcir.shr(%1155 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1157 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out39") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1157 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1156 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1158 = dfcir.add[?] (%1154 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1087 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1159 = dfcir.shr(%1158 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1160 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out31") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1160 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1159 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1161 = dfcir.add[?] (%782 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1153 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1162 = dfcir.sub[?] (%1161 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1111 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1163 = dfcir.shr(%1162 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1164 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out63") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1164 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1163 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1165 = dfcir.add[?] (%1161 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1111 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1166 = dfcir.shr(%1165 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1167 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out7") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1167 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1166 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1168 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 3784 : si32
    %1169 = dfcir.mul[?] (%671 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1168 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1170 = dfcir.sub[?] (%674 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %1169 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>> {latency = -1 : i32}
    %1171 = dfcir.shr(%1170 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1172 = dfcir.sub[?] (%786 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1171 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1173 = dfcir.sub[?] (%1172 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1138 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1174 = dfcir.shr(%1173 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1175 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out40") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1175 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1174 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1176 = dfcir.add[?] (%1172 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1138 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1177 = dfcir.shr(%1176 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1178 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out16") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1178 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1177 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1179 = dfcir.add[?] (%786 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1171 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1180 = dfcir.sub[?] (%1179 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1142 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1181 = dfcir.shr(%1180 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1182 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out48") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1182 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1181 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1183 = dfcir.add[?] (%1179 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1142 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>> {latency = -1 : i32}
    %1184 = dfcir.shr(%1183 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1185 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out8") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1185 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1184 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
  }
}
