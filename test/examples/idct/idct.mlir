module {
  dfcir.kernel "IDCT" {
    %0 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %1 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %2 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %3 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %4 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %5 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %6 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %7 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %8 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %9 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %10 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %11 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %12 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %13 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %14 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %15 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %16 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %17 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %18 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %19 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %20 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %21 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %22 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %23 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %24 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %25 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %26 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %27 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %28 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %29 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x43")
    %30 = dfcir.cast[?] (%29 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %31 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x29")
    %32 = dfcir.cast[?] (%31 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %33 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %34 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x42")
    %35 = dfcir.cast[?] (%34 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %36 = dfcir.mul[?] (%35 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %28 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %37 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x28")
    %38 = dfcir.cast[?] (%37 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %39 = dfcir.shl(%38 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %40 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %41 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x34")
    %42 = dfcir.cast[?] (%41 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %43 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %44 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x33")
    %45 = dfcir.cast[?] (%44 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %46 = dfcir.mul[?] (%45 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %40 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %47 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %48 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x19")
    %49 = dfcir.cast[?] (%48 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %50 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x7")
    %51 = dfcir.cast[?] (%50 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %52 = dfcir.mul[?] (%51 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %33 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %53 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x18")
    %54 = dfcir.cast[?] (%53 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %55 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x0")
    %56 = dfcir.cast[?] (%55 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %57 = dfcir.shl(%56 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %58 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x6")
    %59 = dfcir.cast[?] (%58 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %60 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x40")
    %61 = dfcir.cast[?] (%60 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %62 = dfcir.shl(%61 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %63 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %64 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %65 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %66 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x26")
    %67 = dfcir.cast[?] (%66 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %68 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1108 : si32
    %69 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x11")
    %70 = dfcir.cast[?] (%69 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %71 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x36")
    %72 = dfcir.cast[?] (%71 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %73 = dfcir.shl(%72 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %74 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x22")
    %75 = dfcir.cast[?] (%74 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %76 = dfcir.add[?] (%54 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %75 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %77 = dfcir.mul[?] (%76 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %78 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x9")
    %79 = dfcir.cast[?] (%78 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %80 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x60")
    %81 = dfcir.cast[?] (%80 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %82 = dfcir.shl(%81 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %83 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x10")
    %84 = dfcir.cast[?] (%83 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %85 = dfcir.mul[?] (%84 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %63 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %86 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %87 = dfcir.mul[?] (%54 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %86 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %88 = dfcir.add[?] (%77 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %87 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %89 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1609 : si32
    %90 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x2")
    %91 = dfcir.cast[?] (%90 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %92 = dfcir.add[?] (%91 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %59 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %93 = dfcir.mul[?] (%92 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %94 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x47")
    %95 = dfcir.cast[?] (%94 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %96 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x21")
    %97 = dfcir.cast[?] (%96 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %98 = dfcir.add[?] (%97 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %49 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %99 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2676 : si32
    %100 = dfcir.constant<!dfcir.fixed<true, 15, 0>> 255 : si16
    %101 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %102 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %103 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x35")
    %104 = dfcir.cast[?] (%103 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %105 = dfcir.mul[?] (%104 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %64 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %106 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %107 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x1")
    %108 = dfcir.cast[?] (%107 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %109 = dfcir.add[?] (%108 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %51 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %110 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %111 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x20")
    %112 = dfcir.cast[?] (%111 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %113 = dfcir.shl(%112 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %114 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2841 : si32
    %115 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x37")
    %116 = dfcir.cast[?] (%115 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %117 = dfcir.add[?] (%116 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %104 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %118 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %119 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x23")
    %120 = dfcir.cast[?] (%119 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %121 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x61")
    %122 = dfcir.cast[?] (%121 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %123 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %124 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4 : si32
    %125 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 8192 : si32
    %126 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x32")
    %127 = dfcir.cast[?] (%126 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %128 = dfcir.shl(%127 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %129 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x59")
    %130 = dfcir.cast[?] (%129 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %131 = dfcir.add[?] (%122 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %130 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %132 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 181 : si32
    %133 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2408 : si32
    %134 = dfcir.mul[?] (%131 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %135 = dfcir.mul[?] (%117 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %136 = dfcir.sub[?] (%135 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %105 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %137 = dfcir.mul[?] (%98 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %138 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x55")
    %139 = dfcir.cast[?] (%138 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %140 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x44")
    %141 = dfcir.cast[?] (%140 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %142 = dfcir.shl(%141 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %143 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %144 = dfcir.mul[?] (%95 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %143 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %145 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x38")
    %146 = dfcir.cast[?] (%145 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %147 = dfcir.add[?] (%42 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %146 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %148 = dfcir.mul[?] (%147 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %149 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %150 = dfcir.mul[?] (%130 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %149 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %151 = dfcir.sub[?] (%134 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %150 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %152 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x24")
    %153 = dfcir.cast[?] (%152 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %154 = dfcir.shl(%153 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %155 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 128 : si32
    %156 = dfcir.add[?] (%62 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %157 = dfcir.sub[?] (%156 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %142 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %158 = dfcir.add[?] (%156 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %142 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %159 = dfcir.add[?] (%128 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %160 = dfcir.sub[?] (%159 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %73 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %161 = dfcir.add[?] (%159 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %73 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %162 = dfcir.add[?] (%154 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %163 = dfcir.sub[?] (%162 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %39 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %164 = dfcir.add[?] (%162 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %39 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %165 = dfcir.add[?] (%57 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %166 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x62")
    %167 = dfcir.cast[?] (%166 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %168 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %169 = dfcir.mul[?] (%167 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %168 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %170 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x17")
    %171 = dfcir.cast[?] (%170 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %172 = dfcir.add[?] (%171 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %120 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %173 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %174 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %175 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x41")
    %176 = dfcir.cast[?] (%175 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %177 = dfcir.add[?] (%176 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %95 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %178 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %179 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %180 = dfcir.mul[?] (%67 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %179 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %181 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x27")
    %182 = dfcir.cast[?] (%181 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %183 = dfcir.add[?] (%32 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %182 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %184 = dfcir.mul[?] (%183 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %185 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x3")
    %186 = dfcir.cast[?] (%185 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %187 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x8")
    %188 = dfcir.cast[?] (%187 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %189 = dfcir.shl(%188 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %190 = dfcir.add[?] (%189 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %191 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 565 : si32
    %192 = dfcir.mul[?] (%177 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %193 = dfcir.sub[?] (%192 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %144 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %194 = dfcir.mul[?] (%172 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %195 = dfcir.mul[?] (%109 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %196 = dfcir.sub[?] (%195 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %52 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %197 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %198 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %199 = dfcir.mul[?] (%49 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %198 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %200 = dfcir.sub[?] (%137 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %199 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %201 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x45")
    %202 = dfcir.cast[?] (%201 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %203 = dfcir.mul[?] (%202 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %65 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %204 = dfcir.add[?] (%202 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %30 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %205 = dfcir.mul[?] (%204 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %206 = dfcir.sub[?] (%205 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %203 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %207 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %208 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x39")
    %209 = dfcir.cast[?] (%208 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %210 = dfcir.add[?] (%45 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %209 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %211 = dfcir.mul[?] (%210 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %212 = dfcir.add[?] (%211 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %46 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %213 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %214 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x25")
    %215 = dfcir.cast[?] (%214 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %216 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x63")
    %217 = dfcir.cast[?] (%216 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %218 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x13")
    %219 = dfcir.cast[?] (%218 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %220 = dfcir.add[?] (%219 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %70 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %221 = dfcir.mul[?] (%220 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %222 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %223 = dfcir.mul[?] (%70 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %222 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %224 = dfcir.sub[?] (%221 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %223 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %225 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %226 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x14")
    %227 = dfcir.cast[?] (%226 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %228 = dfcir.mul[?] (%227 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %101 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %229 = dfcir.add[?] (%84 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %227 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %230 = dfcir.mul[?] (%229 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %231 = dfcir.add[?] (%230 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %85 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %232 = dfcir.sub[?] (%230 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %228 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %233 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %234 = dfcir.mul[?] (%75 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %233 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %235 = dfcir.sub[?] (%77 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %234 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %236 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %237 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x5")
    %238 = dfcir.cast[?] (%237 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %239 = dfcir.add[?] (%238 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %186 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %240 = dfcir.mul[?] (%239 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %241 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x15")
    %242 = dfcir.cast[?] (%241 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %243 = dfcir.add[?] (%79 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %242 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %244 = dfcir.mul[?] (%243 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %245 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %246 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x16")
    %247 = dfcir.cast[?] (%246 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %248 = dfcir.shl(%247 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %249 = dfcir.add[?] (%248 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %250 = dfcir.sub[?] (%249 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %113 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %251 = dfcir.sub[?] (%250 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %235 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %252 = dfcir.add[?] (%250 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %235 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %253 = dfcir.add[?] (%249 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %113 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %254 = dfcir.sub[?] (%253 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %88 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %255 = dfcir.add[?] (%253 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %88 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %256 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %257 = dfcir.mul[?] (%209 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %256 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %258 = dfcir.sub[?] (%211 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %257 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %259 = dfcir.sub[?] (%258 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %136 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %260 = dfcir.add[?] (%258 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %136 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %261 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x48")
    %262 = dfcir.cast[?] (%261 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %263 = dfcir.shl(%262 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %264 = dfcir.add[?] (%263 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %265 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x49")
    %266 = dfcir.cast[?] (%265 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %267 = dfcir.add[?] (%266 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %139 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %268 = dfcir.mul[?] (%267 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %269 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %270 = dfcir.mul[?] (%186 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %269 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %271 = dfcir.sub[?] (%240 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %270 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %272 = dfcir.sub[?] (%196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %271 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %273 = dfcir.add[?] (%196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %271 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %274 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %275 = dfcir.mul[?] (%116 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %274 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %276 = dfcir.sub[?] (%135 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %275 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %277 = dfcir.sub[?] (%212 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %276 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %278 = dfcir.sub[?] (%277 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %259 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %279 = dfcir.mul[?] (%278 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %280 = dfcir.add[?] (%279 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %281 = dfcir.shr(%280 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %282 = dfcir.add[?] (%277 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %259 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %283 = dfcir.mul[?] (%282 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %284 = dfcir.add[?] (%283 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %285 = dfcir.shr(%284 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %286 = dfcir.add[?] (%212 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %276 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %287 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %288 = dfcir.mul[?] (%238 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %287 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %289 = dfcir.sub[?] (%240 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %288 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %290 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x50")
    %291 = dfcir.cast[?] (%290 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %292 = dfcir.mul[?] (%291 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %173 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %293 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %294 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %295 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x51")
    %296 = dfcir.cast[?] (%295 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %297 = dfcir.mul[?] (%296 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %245 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %298 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %299 = dfcir.mul[?] (%42 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %298 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %300 = dfcir.add[?] (%148 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %299 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %301 = dfcir.sub[?] (%161 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %302 = dfcir.sub[?] (%301 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %260 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %303 = dfcir.shr(%302 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %304 = dfcir.cast[?] (%303 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %305 = dfcir.cast[?] (%304 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %306 = dfcir.shl(%305 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %307 = dfcir.add[?] (%301 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %260 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %308 = dfcir.shr(%307 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %309 = dfcir.cast[?] (%308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %310 = dfcir.cast[?] (%309 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %311 = dfcir.shl(%310 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %312 = dfcir.add[?] (%161 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %313 = dfcir.sub[?] (%312 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %286 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %314 = dfcir.shr(%313 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %315 = dfcir.cast[?] (%314 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %316 = dfcir.cast[?] (%315 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %317 = dfcir.shl(%316 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %318 = dfcir.add[?] (%312 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %286 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %319 = dfcir.shr(%318 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %320 = dfcir.cast[?] (%319 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %321 = dfcir.cast[?] (%320 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %322 = dfcir.shl(%321 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %323 = dfcir.constant<!dfcir.fixed<true, 15, 0>> -256 : si16
    %324 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %325 = dfcir.mul[?] (%139 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %324 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %326 = dfcir.sub[?] (%268 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %325 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %327 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x52")
    %328 = dfcir.cast[?] (%327 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %329 = dfcir.shl(%328 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %330 = dfcir.sub[?] (%264 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %329 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %331 = dfcir.add[?] (%264 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %329 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %332 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %333 = dfcir.mul[?] (%79 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %332 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %334 = dfcir.add[?] (%244 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %333 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %335 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x53")
    %336 = dfcir.cast[?] (%335 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %337 = dfcir.add[?] (%336 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %296 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %338 = dfcir.mul[?] (%337 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %339 = dfcir.sub[?] (%338 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %297 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %340 = dfcir.sub[?] (%326 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %339 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %341 = dfcir.add[?] (%326 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %339 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %342 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %343 = dfcir.mul[?] (%176 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %342 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %344 = dfcir.add[?] (%192 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %343 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %345 = dfcir.sub[?] (%344 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %206 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %346 = dfcir.add[?] (%344 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %206 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %347 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %348 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x54")
    %349 = dfcir.cast[?] (%348 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %350 = dfcir.mul[?] (%349 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %102 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %351 = dfcir.add[?] (%291 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %349 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %352 = dfcir.mul[?] (%351 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %353 = dfcir.add[?] (%352 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %292 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %354 = dfcir.sub[?] (%331 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %353 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %355 = dfcir.sub[?] (%354 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %341 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %356 = dfcir.shr(%355 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %357 = dfcir.cast[?] (%356 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %358 = dfcir.cast[?] (%357 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %359 = dfcir.mul[?] (%358 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %225 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %360 = dfcir.add[?] (%354 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %341 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %361 = dfcir.shr(%360 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %362 = dfcir.cast[?] (%361 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %363 = dfcir.cast[?] (%362 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %364 = dfcir.mul[?] (%363 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %47 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %365 = dfcir.add[?] (%331 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %353 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %366 = dfcir.sub[?] (%352 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %350 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %367 = dfcir.sub[?] (%330 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %366 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %368 = dfcir.add[?] (%330 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %366 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %369 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x56")
    %370 = dfcir.cast[?] (%369 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %371 = dfcir.shl(%370 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %372 = dfcir.add[?] (%371 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %373 = dfcir.sub[?] (%372 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %82 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %374 = dfcir.add[?] (%372 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %82 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %375 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x30")
    %376 = dfcir.cast[?] (%375 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %377 = dfcir.mul[?] (%376 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %293 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %378 = dfcir.add[?] (%67 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %376 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %379 = dfcir.mul[?] (%378 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %380 = dfcir.add[?] (%379 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %180 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %381 = dfcir.sub[?] (%164 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %380 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %382 = dfcir.add[?] (%164 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %380 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %383 = dfcir.sub[?] (%379 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %377 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %384 = dfcir.sub[?] (%163 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %383 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %385 = dfcir.add[?] (%163 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %383 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %386 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x57")
    %387 = dfcir.cast[?] (%386 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %388 = dfcir.add[?] (%387 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %217 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %389 = dfcir.mul[?] (%388 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %390 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x31")
    %391 = dfcir.cast[?] (%390 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %392 = dfcir.mul[?] (%391 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %43 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %393 = dfcir.add[?] (%215 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %391 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %394 = dfcir.mul[?] (%393 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %395 = dfcir.sub[?] (%394 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %392 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %396 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x58")
    %397 = dfcir.cast[?] (%396 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %398 = dfcir.add[?] (%397 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %167 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %399 = dfcir.mul[?] (%398 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %400 = dfcir.sub[?] (%399 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %169 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %401 = dfcir.sub[?] (%373 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %400 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %402 = dfcir.add[?] (%373 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %400 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %403 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %404 = dfcir.mul[?] (%397 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %403 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %405 = dfcir.add[?] (%399 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %404 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %406 = dfcir.sub[?] (%374 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %405 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %407 = dfcir.add[?] (%374 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %405 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %408 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %409 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %410 = dfcir.mul[?] (%108 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %409 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %411 = dfcir.add[?] (%195 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %410 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %412 = dfcir.sub[?] (%411 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %289 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %413 = dfcir.sub[?] (%412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %272 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %414 = dfcir.mul[?] (%413 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %415 = dfcir.add[?] (%414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %416 = dfcir.shr(%415 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %417 = dfcir.add[?] (%412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %272 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %418 = dfcir.mul[?] (%417 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %419 = dfcir.add[?] (%418 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %420 = dfcir.shr(%419 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %421 = dfcir.add[?] (%411 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %289 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %422 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x4")
    %423 = dfcir.cast[?] (%422 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %424 = dfcir.shl(%423 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %425 = dfcir.sub[?] (%165 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %424 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %426 = dfcir.add[?] (%165 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %424 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %427 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %428 = dfcir.mul[?] (%59 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %427 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %429 = dfcir.sub[?] (%93 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %428 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %430 = dfcir.sub[?] (%425 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %429 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %431 = dfcir.sub[?] (%430 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %432 = dfcir.shr(%431 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %433 = dfcir.cast[?] (%432 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %434 = dfcir.cast[?] (%433 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %435 = dfcir.shl(%434 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %436 = dfcir.add[?] (%435 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %437 = dfcir.add[?] (%430 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %438 = dfcir.shr(%437 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %439 = dfcir.cast[?] (%438 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %440 = dfcir.cast[?] (%439 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %441 = dfcir.shl(%440 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %442 = dfcir.add[?] (%441 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %443 = dfcir.add[?] (%425 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %429 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %444 = dfcir.sub[?] (%443 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %420 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %445 = dfcir.shr(%444 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %446 = dfcir.cast[?] (%445 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %447 = dfcir.cast[?] (%446 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %448 = dfcir.shl(%447 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %449 = dfcir.add[?] (%448 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %450 = dfcir.add[?] (%443 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %420 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %451 = dfcir.shr(%450 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %452 = dfcir.cast[?] (%451 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %453 = dfcir.cast[?] (%452 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %454 = dfcir.shl(%453 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %455 = dfcir.add[?] (%454 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %456 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %457 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %458 = dfcir.mul[?] (%91 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %457 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %459 = dfcir.add[?] (%93 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %458 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %460 = dfcir.sub[?] (%426 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %459 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %461 = dfcir.sub[?] (%460 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %273 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %462 = dfcir.shr(%461 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %463 = dfcir.cast[?] (%462 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %464 = dfcir.cast[?] (%463 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %465 = dfcir.shl(%464 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %466 = dfcir.add[?] (%465 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %467 = dfcir.sub[?] (%466 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %306 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %468 = dfcir.add[?] (%466 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %306 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %469 = dfcir.add[?] (%460 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %273 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %470 = dfcir.shr(%469 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %471 = dfcir.cast[?] (%470 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %472 = dfcir.cast[?] (%471 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %473 = dfcir.shl(%472 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %474 = dfcir.add[?] (%473 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %475 = dfcir.sub[?] (%474 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %311 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %476 = dfcir.add[?] (%474 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %311 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %477 = dfcir.add[?] (%426 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %459 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %478 = dfcir.sub[?] (%477 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %421 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %479 = dfcir.shr(%478 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %480 = dfcir.cast[?] (%479 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %481 = dfcir.cast[?] (%480 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %482 = dfcir.shl(%481 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %483 = dfcir.add[?] (%482 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %484 = dfcir.sub[?] (%483 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %317 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %485 = dfcir.add[?] (%483 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %317 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %486 = dfcir.add[?] (%477 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %421 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %487 = dfcir.shr(%486 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %488 = dfcir.cast[?] (%487 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %489 = dfcir.cast[?] (%488 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %490 = dfcir.shl(%489 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %491 = dfcir.add[?] (%490 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %492 = dfcir.sub[?] (%491 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %322 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %493 = dfcir.add[?] (%491 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %322 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %494 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x46")
    %495 = dfcir.cast[?] (%494 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %496 = dfcir.add[?] (%35 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %495 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %497 = dfcir.mul[?] (%496 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %498 = dfcir.add[?] (%497 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %36 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %499 = dfcir.sub[?] (%158 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %498 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %500 = dfcir.add[?] (%158 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %498 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %501 = dfcir.sub[?] (%500 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %346 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %502 = dfcir.shr(%501 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %503 = dfcir.cast[?] (%502 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %504 = dfcir.cast[?] (%503 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %505 = dfcir.mul[?] (%504 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %506 = dfcir.add[?] (%500 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %346 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %507 = dfcir.shr(%506 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %508 = dfcir.cast[?] (%507 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %509 = dfcir.cast[?] (%508 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %510 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %511 = dfcir.mul[?] (%242 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %510 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %512 = dfcir.sub[?] (%244 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %511 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %513 = dfcir.sub[?] (%512 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %224 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %514 = dfcir.add[?] (%512 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %224 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %515 = dfcir.input<!dfcir.fixed<true, 15, 0>> ("x12")
    %516 = dfcir.cast[?] (%515 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %517 = dfcir.shl(%516 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %518 = dfcir.sub[?] (%190 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %517 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %519 = dfcir.sub[?] (%518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %232 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %520 = dfcir.add[?] (%518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %232 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %521 = dfcir.add[?] (%190 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %517 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %522 = dfcir.sub[?] (%521 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %231 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %523 = dfcir.sub[?] (%522 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %514 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %524 = dfcir.shr(%523 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %525 = dfcir.cast[?] (%524 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %526 = dfcir.cast[?] (%525 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %527 = dfcir.mul[?] (%526 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %528 = dfcir.add[?] (%522 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %514 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %529 = dfcir.shr(%528 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %530 = dfcir.cast[?] (%529 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %531 = dfcir.cast[?] (%530 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %532 = dfcir.mul[?] (%531 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %16 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %533 = dfcir.add[?] (%521 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %231 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %534 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %535 = dfcir.mul[?] (%219 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %534 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %536 = dfcir.sub[?] (%221 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %535 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %537 = dfcir.sub[?] (%334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %536 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %538 = dfcir.sub[?] (%537 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %513 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %539 = dfcir.mul[?] (%538 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %540 = dfcir.add[?] (%539 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %541 = dfcir.shr(%540 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %542 = dfcir.sub[?] (%519 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %541 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %543 = dfcir.shr(%542 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %544 = dfcir.cast[?] (%543 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %545 = dfcir.cast[?] (%544 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %546 = dfcir.mul[?] (%545 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %347 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %547 = dfcir.add[?] (%519 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %541 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %548 = dfcir.shr(%547 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %549 = dfcir.cast[?] (%548 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %550 = dfcir.cast[?] (%549 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %551 = dfcir.mul[?] (%550 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %21 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %552 = dfcir.add[?] (%537 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %513 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %553 = dfcir.mul[?] (%552 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %554 = dfcir.add[?] (%553 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %555 = dfcir.shr(%554 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %556 = dfcir.sub[?] (%520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %555 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %557 = dfcir.shr(%556 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %558 = dfcir.cast[?] (%557 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %559 = dfcir.cast[?] (%558 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %560 = dfcir.mul[?] (%559 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %106 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %561 = dfcir.add[?] (%520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %555 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %562 = dfcir.shr(%561 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %563 = dfcir.cast[?] (%562 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %564 = dfcir.cast[?] (%563 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %565 = dfcir.mul[?] (%564 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %25 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %566 = dfcir.add[?] (%334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %536 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %567 = dfcir.sub[?] (%533 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %566 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %568 = dfcir.shr(%567 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %569 = dfcir.cast[?] (%568 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %570 = dfcir.cast[?] (%569 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %571 = dfcir.mul[?] (%570 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %572 = dfcir.add[?] (%533 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %566 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %573 = dfcir.shr(%572 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %574 = dfcir.cast[?] (%573 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %575 = dfcir.cast[?] (%574 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %576 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %577 = dfcir.mul[?] (%266 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %576 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %578 = dfcir.add[?] (%268 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %577 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %579 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %580 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %581 = dfcir.mul[?] (%171 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %580 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %582 = dfcir.add[?] (%194 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %581 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %583 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %584 = dfcir.mul[?] (%120 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %583 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %585 = dfcir.sub[?] (%194 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %584 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %586 = dfcir.sub[?] (%585 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %587 = dfcir.add[?] (%585 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %588 = dfcir.sub[?] (%254 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %587 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %589 = dfcir.shr(%588 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %590 = dfcir.cast[?] (%589 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %591 = dfcir.cast[?] (%590 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %592 = dfcir.mul[?] (%591 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %207 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %593 = dfcir.add[?] (%591 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %358 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %594 = dfcir.mul[?] (%593 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %595 = dfcir.add[?] (%594 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %596 = dfcir.add[?] (%595 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %592 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %597 = dfcir.shr(%596 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %598 = dfcir.sub[?] (%468 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %597 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %599 = dfcir.add[?] (%468 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %597 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %600 = dfcir.sub[?] (%595 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %359 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %601 = dfcir.shr(%600 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %602 = dfcir.sub[?] (%467 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %601 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %603 = dfcir.add[?] (%467 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %601 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %604 = dfcir.add[?] (%254 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %587 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %605 = dfcir.shr(%604 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %606 = dfcir.cast[?] (%605 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %607 = dfcir.cast[?] (%606 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %608 = dfcir.mul[?] (%607 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %609 = dfcir.add[?] (%607 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %363 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %610 = dfcir.mul[?] (%609 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %611 = dfcir.add[?] (%610 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %612 = dfcir.add[?] (%611 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %608 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %613 = dfcir.shr(%612 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %614 = dfcir.sub[?] (%476 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %613 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %615 = dfcir.add[?] (%476 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %613 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %616 = dfcir.sub[?] (%611 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %364 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %617 = dfcir.shr(%616 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %618 = dfcir.sub[?] (%475 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %617 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %619 = dfcir.add[?] (%475 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %617 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %620 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %621 = dfcir.mul[?] (%336 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %620 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %622 = dfcir.sub[?] (%338 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %621 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %623 = dfcir.sub[?] (%578 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %622 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %624 = dfcir.sub[?] (%623 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %340 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %625 = dfcir.mul[?] (%624 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %626 = dfcir.add[?] (%625 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %627 = dfcir.shr(%626 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %628 = dfcir.sub[?] (%367 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %627 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %629 = dfcir.shr(%628 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %630 = dfcir.cast[?] (%629 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %631 = dfcir.cast[?] (%630 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %632 = dfcir.mul[?] (%631 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %123 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %633 = dfcir.add[?] (%367 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %627 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %634 = dfcir.shr(%633 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %635 = dfcir.cast[?] (%634 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %636 = dfcir.cast[?] (%635 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %637 = dfcir.mul[?] (%636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %408 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %638 = dfcir.add[?] (%623 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %340 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %639 = dfcir.mul[?] (%638 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %640 = dfcir.add[?] (%639 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %641 = dfcir.shr(%640 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %642 = dfcir.sub[?] (%368 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %641 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %643 = dfcir.shr(%642 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %644 = dfcir.cast[?] (%643 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %645 = dfcir.cast[?] (%644 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %646 = dfcir.mul[?] (%645 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %647 = dfcir.add[?] (%368 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %641 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %648 = dfcir.shr(%647 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %649 = dfcir.cast[?] (%648 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %650 = dfcir.cast[?] (%649 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %651 = dfcir.mul[?] (%650 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %23 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %652 = dfcir.add[?] (%578 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %622 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %653 = dfcir.sub[?] (%365 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %652 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %654 = dfcir.shr(%653 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %655 = dfcir.cast[?] (%654 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %656 = dfcir.cast[?] (%655 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %657 = dfcir.mul[?] (%656 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %658 = dfcir.add[?] (%365 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %652 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %659 = dfcir.shr(%658 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %660 = dfcir.cast[?] (%659 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %661 = dfcir.cast[?] (%660 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %662 = dfcir.mul[?] (%661 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %26 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %663 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %664 = dfcir.mul[?] (%97 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %663 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %665 = dfcir.sub[?] (%137 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %664 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %666 = dfcir.sub[?] (%582 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %665 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %667 = dfcir.sub[?] (%666 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %586 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %668 = dfcir.mul[?] (%667 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %669 = dfcir.add[?] (%668 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %670 = dfcir.shr(%669 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %671 = dfcir.sub[?] (%251 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %670 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %672 = dfcir.shr(%671 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %673 = dfcir.cast[?] (%672 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %674 = dfcir.cast[?] (%673 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %675 = dfcir.mul[?] (%674 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %676 = dfcir.add[?] (%674 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %631 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %677 = dfcir.mul[?] (%676 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %678 = dfcir.add[?] (%677 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %679 = dfcir.add[?] (%678 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %675 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %680 = dfcir.shr(%679 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %681 = dfcir.sub[?] (%678 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %632 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %682 = dfcir.shr(%681 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %683 = dfcir.add[?] (%251 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %670 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %684 = dfcir.shr(%683 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %685 = dfcir.cast[?] (%684 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %686 = dfcir.cast[?] (%685 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %687 = dfcir.mul[?] (%686 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %17 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %688 = dfcir.add[?] (%686 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %689 = dfcir.mul[?] (%688 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %690 = dfcir.add[?] (%689 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %691 = dfcir.add[?] (%690 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %687 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %692 = dfcir.shr(%691 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %693 = dfcir.sub[?] (%690 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %637 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %694 = dfcir.shr(%693 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %695 = dfcir.add[?] (%666 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %586 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %696 = dfcir.mul[?] (%695 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %697 = dfcir.add[?] (%696 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %698 = dfcir.shr(%697 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %699 = dfcir.sub[?] (%252 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %698 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %700 = dfcir.shr(%699 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %701 = dfcir.cast[?] (%700 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %702 = dfcir.cast[?] (%701 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %703 = dfcir.mul[?] (%702 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %174 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %704 = dfcir.add[?] (%702 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %645 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %705 = dfcir.mul[?] (%704 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %706 = dfcir.add[?] (%705 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %707 = dfcir.add[?] (%706 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %703 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %708 = dfcir.shr(%707 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %709 = dfcir.sub[?] (%706 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %646 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %710 = dfcir.shr(%709 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %711 = dfcir.add[?] (%252 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %698 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %712 = dfcir.shr(%711 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %713 = dfcir.cast[?] (%712 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %714 = dfcir.cast[?] (%713 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %715 = dfcir.mul[?] (%714 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %22 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %716 = dfcir.add[?] (%714 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %650 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %717 = dfcir.mul[?] (%716 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %718 = dfcir.add[?] (%717 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %719 = dfcir.add[?] (%718 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %715 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %720 = dfcir.shr(%719 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %721 = dfcir.sub[?] (%718 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %651 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %722 = dfcir.shr(%721 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %723 = dfcir.add[?] (%582 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %665 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %724 = dfcir.sub[?] (%255 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %723 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %725 = dfcir.shr(%724 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %726 = dfcir.cast[?] (%725 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %727 = dfcir.cast[?] (%726 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %728 = dfcir.mul[?] (%727 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %729 = dfcir.add[?] (%727 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %656 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %730 = dfcir.mul[?] (%729 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %731 = dfcir.add[?] (%730 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %732 = dfcir.add[?] (%731 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %728 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %733 = dfcir.shr(%732 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %734 = dfcir.sub[?] (%485 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %733 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %735 = dfcir.add[?] (%485 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %733 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %736 = dfcir.sub[?] (%731 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %657 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %737 = dfcir.shr(%736 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %738 = dfcir.sub[?] (%484 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %737 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %739 = dfcir.add[?] (%484 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %737 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %740 = dfcir.add[?] (%255 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %723 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %741 = dfcir.shr(%740 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %742 = dfcir.cast[?] (%741 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %743 = dfcir.cast[?] (%742 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %744 = dfcir.mul[?] (%743 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %456 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %745 = dfcir.add[?] (%743 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %661 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %746 = dfcir.mul[?] (%745 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %68 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %747 = dfcir.add[?] (%746 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %748 = dfcir.add[?] (%747 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %744 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %749 = dfcir.shr(%748 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %750 = dfcir.sub[?] (%493 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %749 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %751 = dfcir.add[?] (%493 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %749 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %752 = dfcir.sub[?] (%747 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %662 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %753 = dfcir.shr(%752 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %754 = dfcir.sub[?] (%492 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %753 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %755 = dfcir.add[?] (%492 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %753 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %756 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %757 = dfcir.mul[?] (%215 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %756 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %758 = dfcir.add[?] (%394 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %757 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %759 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %760 = dfcir.mul[?] (%217 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %759 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %761 = dfcir.sub[?] (%389 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %760 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %762 = dfcir.sub[?] (%761 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %151 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %763 = dfcir.add[?] (%761 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %151 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %764 = dfcir.sub[?] (%406 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %763 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %765 = dfcir.shr(%764 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %766 = dfcir.cast[?] (%765 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %767 = dfcir.cast[?] (%766 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %768 = dfcir.mul[?] (%767 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %118 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %769 = dfcir.add[?] (%526 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %767 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %770 = dfcir.mul[?] (%769 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %771 = dfcir.add[?] (%770 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %772 = dfcir.sub[?] (%771 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %768 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %773 = dfcir.shr(%772 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %774 = dfcir.add[?] (%771 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %527 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %775 = dfcir.shr(%774 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %776 = dfcir.add[?] (%406 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %763 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %777 = dfcir.shr(%776 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %778 = dfcir.cast[?] (%777 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %779 = dfcir.cast[?] (%778 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %780 = dfcir.mul[?] (%779 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %178 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %781 = dfcir.add[?] (%531 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %779 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %782 = dfcir.mul[?] (%781 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %783 = dfcir.add[?] (%782 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %784 = dfcir.sub[?] (%783 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %780 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %785 = dfcir.shr(%784 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %786 = dfcir.add[?] (%783 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %532 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %787 = dfcir.shr(%786 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %788 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %789 = dfcir.mul[?] (%182 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %788 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %790 = dfcir.sub[?] (%184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %789 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %791 = dfcir.sub[?] (%395 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %790 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %792 = dfcir.add[?] (%395 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %790 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %793 = dfcir.sub[?] (%381 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %792 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %794 = dfcir.shr(%793 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %795 = dfcir.cast[?] (%794 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %796 = dfcir.cast[?] (%795 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %797 = dfcir.mul[?] (%796 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %798 = dfcir.add[?] (%381 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %792 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %799 = dfcir.shr(%798 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %800 = dfcir.cast[?] (%799 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %801 = dfcir.cast[?] (%800 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %802 = dfcir.mul[?] (%801 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %27 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %803 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %804 = dfcir.mul[?] (%146 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %803 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %805 = dfcir.sub[?] (%148 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %804 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %806 = dfcir.sub[?] (%160 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %805 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %807 = dfcir.sub[?] (%806 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %281 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %808 = dfcir.shr(%807 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %809 = dfcir.cast[?] (%808 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %810 = dfcir.cast[?] (%809 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %811 = dfcir.shl(%810 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %812 = dfcir.sub[?] (%436 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %811 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %813 = dfcir.sub[?] (%812 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %682 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %814 = dfcir.add[?] (%812 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %682 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %815 = dfcir.add[?] (%436 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %811 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %816 = dfcir.sub[?] (%815 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %680 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %817 = dfcir.add[?] (%815 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %680 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %818 = dfcir.add[?] (%806 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %281 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %819 = dfcir.shr(%818 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %820 = dfcir.cast[?] (%819 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %821 = dfcir.cast[?] (%820 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %822 = dfcir.shl(%821 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %823 = dfcir.sub[?] (%442 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %822 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %824 = dfcir.sub[?] (%823 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %694 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %825 = dfcir.add[?] (%823 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %694 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %826 = dfcir.add[?] (%442 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %822 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %827 = dfcir.sub[?] (%826 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %692 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %828 = dfcir.add[?] (%826 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %692 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %829 = dfcir.add[?] (%160 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %805 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %830 = dfcir.sub[?] (%829 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %285 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %831 = dfcir.shr(%830 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %832 = dfcir.cast[?] (%831 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %833 = dfcir.cast[?] (%832 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %834 = dfcir.shl(%833 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %835 = dfcir.sub[?] (%449 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %834 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %836 = dfcir.sub[?] (%835 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %710 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %837 = dfcir.add[?] (%835 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %710 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %838 = dfcir.add[?] (%449 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %834 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %839 = dfcir.sub[?] (%838 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %708 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %840 = dfcir.add[?] (%838 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %708 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %841 = dfcir.add[?] (%829 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %285 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %842 = dfcir.shr(%841 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %843 = dfcir.cast[?] (%842 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %844 = dfcir.cast[?] (%843 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %845 = dfcir.shl(%844 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %846 = dfcir.sub[?] (%455 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %845 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %847 = dfcir.sub[?] (%846 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %722 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %848 = dfcir.add[?] (%846 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %722 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %849 = dfcir.add[?] (%455 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %845 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %850 = dfcir.sub[?] (%849 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %720 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %851 = dfcir.add[?] (%849 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %720 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %852 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %853 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %854 = dfcir.mul[?] (%32 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %853 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %855 = dfcir.sub[?] (%184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %854 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %856 = dfcir.sub[?] (%758 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %855 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %857 = dfcir.sub[?] (%856 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %791 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %858 = dfcir.mul[?] (%857 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %859 = dfcir.add[?] (%858 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %860 = dfcir.shr(%859 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %861 = dfcir.sub[?] (%384 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %860 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %862 = dfcir.shr(%861 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %863 = dfcir.cast[?] (%862 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %864 = dfcir.cast[?] (%863 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %865 = dfcir.mul[?] (%864 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %9 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %866 = dfcir.add[?] (%384 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %860 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %867 = dfcir.shr(%866 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %868 = dfcir.cast[?] (%867 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %869 = dfcir.cast[?] (%868 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %870 = dfcir.mul[?] (%869 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %18 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %871 = dfcir.add[?] (%856 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %791 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %872 = dfcir.mul[?] (%871 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %873 = dfcir.add[?] (%872 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %874 = dfcir.shr(%873 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %875 = dfcir.sub[?] (%385 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %874 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %876 = dfcir.shr(%875 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %877 = dfcir.cast[?] (%876 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %878 = dfcir.cast[?] (%877 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %879 = dfcir.mul[?] (%878 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %236 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %880 = dfcir.add[?] (%385 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %874 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %881 = dfcir.shr(%880 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %882 = dfcir.cast[?] (%881 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %883 = dfcir.cast[?] (%882 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %884 = dfcir.mul[?] (%883 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %197 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %885 = dfcir.add[?] (%758 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %855 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %886 = dfcir.sub[?] (%382 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %885 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %887 = dfcir.shr(%886 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %888 = dfcir.cast[?] (%887 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %889 = dfcir.cast[?] (%888 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %890 = dfcir.mul[?] (%889 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %891 = dfcir.add[?] (%504 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %889 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %892 = dfcir.mul[?] (%891 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %893 = dfcir.add[?] (%892 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %894 = dfcir.sub[?] (%893 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %890 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %895 = dfcir.shr(%894 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %896 = dfcir.sub[?] (%893 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %505 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %897 = dfcir.shr(%896 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %898 = dfcir.add[?] (%382 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %885 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %899 = dfcir.shr(%898 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %900 = dfcir.cast[?] (%899 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %901 = dfcir.cast[?] (%900 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %902 = dfcir.mul[?] (%901 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %852 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %903 = dfcir.add[?] (%509 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %901 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %904 = dfcir.mul[?] (%903 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %905 = dfcir.add[?] (%904 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %906 = dfcir.sub[?] (%905 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %902 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %907 = dfcir.shr(%906 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %908 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %909 = dfcir.mul[?] (%387 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %908 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %910 = dfcir.add[?] (%389 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %909 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %911 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %912 = dfcir.mul[?] (%30 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %911 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %913 = dfcir.sub[?] (%205 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %912 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %914 = dfcir.sub[?] (%193 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %913 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %915 = dfcir.sub[?] (%345 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %914 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %916 = dfcir.mul[?] (%915 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %917 = dfcir.add[?] (%916 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %918 = dfcir.shr(%917 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %919 = dfcir.add[?] (%345 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %914 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %920 = dfcir.mul[?] (%919 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %921 = dfcir.add[?] (%920 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %922 = dfcir.shr(%921 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %923 = dfcir.add[?] (%193 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %913 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %924 = dfcir.sub[?] (%499 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %923 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %925 = dfcir.shr(%924 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %926 = dfcir.cast[?] (%925 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %927 = dfcir.cast[?] (%926 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %928 = dfcir.mul[?] (%927 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %929 = dfcir.add[?] (%927 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %796 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %930 = dfcir.mul[?] (%929 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %931 = dfcir.add[?] (%930 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %932 = dfcir.sub[?] (%931 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %797 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %933 = dfcir.shr(%932 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %934 = dfcir.sub[?] (%773 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %933 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %935 = dfcir.add[?] (%773 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %933 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %936 = dfcir.sub[?] (%598 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %935 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %937 = dfcir.shr(%936 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %938 = dfcir.cast[?] (%937 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %939 = dfcir.greater[?] (%938 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %940 = dfcir.cast[?] (%939 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %941 = dfcir.greaterEq[?] (%938 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %942 = dfcir.cast[?] (%941 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %943 = dfcir.add[?] (%942 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %940 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %944 = dfcir.mux(%943: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %938: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %945 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out36") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%945 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %944 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %946 = dfcir.add[?] (%598 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %935 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %947 = dfcir.shr(%946 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %948 = dfcir.cast[?] (%947 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %949 = dfcir.greater[?] (%948 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %950 = dfcir.cast[?] (%949 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %951 = dfcir.greaterEq[?] (%948 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %952 = dfcir.cast[?] (%951 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %953 = dfcir.add[?] (%952 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %950 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %954 = dfcir.mux(%953: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %948: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %955 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out28") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%955 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %954 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %956 = dfcir.sub[?] (%931 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %928 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %957 = dfcir.shr(%956 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %958 = dfcir.sub[?] (%775 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %957 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %959 = dfcir.sub[?] (%958 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %934 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %960 = dfcir.mul[?] (%959 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %961 = dfcir.add[?] (%960 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %962 = dfcir.shr(%961 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %963 = dfcir.sub[?] (%602 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %962 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %964 = dfcir.shr(%963 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %965 = dfcir.cast[?] (%964 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %966 = dfcir.greater[?] (%965 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %967 = dfcir.cast[?] (%966 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %968 = dfcir.greaterEq[?] (%965 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %969 = dfcir.cast[?] (%968 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %970 = dfcir.add[?] (%969 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %967 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %971 = dfcir.mux(%970: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %965: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %972 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out44") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%972 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %971 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %973 = dfcir.add[?] (%602 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %962 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %974 = dfcir.shr(%973 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %975 = dfcir.cast[?] (%974 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %976 = dfcir.greater[?] (%975 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %977 = dfcir.cast[?] (%976 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %978 = dfcir.greaterEq[?] (%975 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %979 = dfcir.cast[?] (%978 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %980 = dfcir.add[?] (%979 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %977 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %981 = dfcir.mux(%980: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %975: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %982 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out20") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%982 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %981 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %983 = dfcir.add[?] (%958 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %934 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %984 = dfcir.mul[?] (%983 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %985 = dfcir.add[?] (%984 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %986 = dfcir.shr(%985 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %987 = dfcir.sub[?] (%603 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %986 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %988 = dfcir.shr(%987 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %989 = dfcir.cast[?] (%988 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %990 = dfcir.greater[?] (%989 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %991 = dfcir.cast[?] (%990 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %992 = dfcir.greaterEq[?] (%989 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %993 = dfcir.cast[?] (%992 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %994 = dfcir.add[?] (%993 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %991 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %995 = dfcir.mux(%994: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %989: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %996 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out52") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%996 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %995 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %997 = dfcir.add[?] (%603 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %986 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %998 = dfcir.shr(%997 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %999 = dfcir.cast[?] (%998 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1000 = dfcir.greater[?] (%999 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1001 = dfcir.cast[?] (%1000 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1002 = dfcir.greaterEq[?] (%999 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1003 = dfcir.cast[?] (%1002 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1004 = dfcir.add[?] (%1003 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1001 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1005 = dfcir.mux(%1004: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %999: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1006 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out12") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1006 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1005 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1007 = dfcir.add[?] (%775 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %957 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1008 = dfcir.sub[?] (%599 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1007 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1009 = dfcir.shr(%1008 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1010 = dfcir.cast[?] (%1009 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1011 = dfcir.greater[?] (%1010 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1012 = dfcir.cast[?] (%1011 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1013 = dfcir.greaterEq[?] (%1010 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1014 = dfcir.cast[?] (%1013 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1015 = dfcir.add[?] (%1014 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1012 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1016 = dfcir.mux(%1015: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1010: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1017 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out60") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1017 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1016 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1018 = dfcir.add[?] (%599 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1007 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1019 = dfcir.shr(%1018 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1020 = dfcir.cast[?] (%1019 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1021 = dfcir.greater[?] (%1020 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1022 = dfcir.cast[?] (%1021 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1023 = dfcir.greaterEq[?] (%1020 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1024 = dfcir.cast[?] (%1023 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1025 = dfcir.add[?] (%1024 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1022 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1026 = dfcir.mux(%1025: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1020: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1027 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out4") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1027 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1026 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1028 = dfcir.add[?] (%499 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %923 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1029 = dfcir.shr(%1028 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1030 = dfcir.cast[?] (%1029 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1031 = dfcir.cast[?] (%1030 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1032 = dfcir.mul[?] (%1031 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %15 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1033 = dfcir.add[?] (%1031 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %801 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1034 = dfcir.mul[?] (%1033 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1035 = dfcir.add[?] (%1034 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1036 = dfcir.sub[?] (%1035 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %802 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1037 = dfcir.shr(%1036 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1038 = dfcir.sub[?] (%785 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1037 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1039 = dfcir.add[?] (%785 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1037 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1040 = dfcir.sub[?] (%614 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1039 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1041 = dfcir.shr(%1040 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1042 = dfcir.cast[?] (%1041 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1043 = dfcir.greater[?] (%1042 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1044 = dfcir.cast[?] (%1043 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1045 = dfcir.greaterEq[?] (%1042 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1046 = dfcir.cast[?] (%1045 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1047 = dfcir.add[?] (%1046 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1044 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1048 = dfcir.mux(%1047: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1042: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1049 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out35") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1049 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1048 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1050 = dfcir.add[?] (%614 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1039 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1051 = dfcir.shr(%1050 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1052 = dfcir.cast[?] (%1051 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1053 = dfcir.greater[?] (%1052 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1054 = dfcir.cast[?] (%1053 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1055 = dfcir.greaterEq[?] (%1052 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1056 = dfcir.cast[?] (%1055 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1057 = dfcir.add[?] (%1056 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1054 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1058 = dfcir.mux(%1057: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1052: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1059 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out27") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1059 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1058 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1060 = dfcir.sub[?] (%1035 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1032 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1061 = dfcir.shr(%1060 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1062 = dfcir.sub[?] (%787 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1061 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1063 = dfcir.sub[?] (%1062 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1038 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1064 = dfcir.mul[?] (%1063 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1065 = dfcir.add[?] (%1064 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1066 = dfcir.shr(%1065 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1067 = dfcir.sub[?] (%618 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1066 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1068 = dfcir.shr(%1067 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1069 = dfcir.cast[?] (%1068 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1070 = dfcir.greater[?] (%1069 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1071 = dfcir.cast[?] (%1070 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1072 = dfcir.greaterEq[?] (%1069 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1073 = dfcir.cast[?] (%1072 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1074 = dfcir.add[?] (%1073 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1071 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1075 = dfcir.mux(%1074: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1069: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1076 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out43") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1076 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1075 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1077 = dfcir.add[?] (%618 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1066 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1078 = dfcir.shr(%1077 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1079 = dfcir.cast[?] (%1078 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1080 = dfcir.greater[?] (%1079 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1081 = dfcir.cast[?] (%1080 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1082 = dfcir.greaterEq[?] (%1079 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1083 = dfcir.cast[?] (%1082 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1084 = dfcir.add[?] (%1083 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1081 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1085 = dfcir.mux(%1084: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1079: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1086 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out19") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1086 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1085 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1087 = dfcir.add[?] (%1062 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1038 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1088 = dfcir.mul[?] (%1087 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1089 = dfcir.add[?] (%1088 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1090 = dfcir.shr(%1089 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1091 = dfcir.sub[?] (%619 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1090 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1092 = dfcir.shr(%1091 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1093 = dfcir.cast[?] (%1092 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1094 = dfcir.greater[?] (%1093 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1095 = dfcir.cast[?] (%1094 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1096 = dfcir.greaterEq[?] (%1093 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1097 = dfcir.cast[?] (%1096 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1098 = dfcir.add[?] (%1097 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1095 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1099 = dfcir.mux(%1098: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1093: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1100 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out51") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1100 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1099 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1101 = dfcir.add[?] (%619 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1090 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1102 = dfcir.shr(%1101 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1103 = dfcir.cast[?] (%1102 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1104 = dfcir.greater[?] (%1103 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1105 = dfcir.cast[?] (%1104 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1106 = dfcir.greaterEq[?] (%1103 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1107 = dfcir.cast[?] (%1106 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1108 = dfcir.add[?] (%1107 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1105 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1109 = dfcir.mux(%1108: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1103: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1110 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out11") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1110 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1109 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1111 = dfcir.add[?] (%787 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1061 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1112 = dfcir.sub[?] (%615 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1111 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1113 = dfcir.shr(%1112 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1114 = dfcir.cast[?] (%1113 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1115 = dfcir.greater[?] (%1114 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1116 = dfcir.cast[?] (%1115 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1117 = dfcir.greaterEq[?] (%1114 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1118 = dfcir.cast[?] (%1117 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1119 = dfcir.add[?] (%1118 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1116 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1120 = dfcir.mux(%1119: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1114: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1121 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out59") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1121 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1120 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1122 = dfcir.add[?] (%615 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1111 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1123 = dfcir.shr(%1122 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1124 = dfcir.cast[?] (%1123 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1125 = dfcir.greater[?] (%1124 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1126 = dfcir.cast[?] (%1125 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1127 = dfcir.greaterEq[?] (%1124 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1128 = dfcir.cast[?] (%1127 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1129 = dfcir.add[?] (%1128 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1126 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1130 = dfcir.mux(%1129: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1124: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1131 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out3") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1131 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1130 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1132 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %1133 = dfcir.mul[?] (%495 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1134 = dfcir.sub[?] (%497 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1133 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1135 = dfcir.sub[?] (%157 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1134 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1136 = dfcir.sub[?] (%1135 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %918 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1137 = dfcir.shr(%1136 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1138 = dfcir.cast[?] (%1137 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1139 = dfcir.cast[?] (%1138 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1140 = dfcir.mul[?] (%1139 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %294 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1141 = dfcir.add[?] (%1139 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %864 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1142 = dfcir.mul[?] (%1141 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1143 = dfcir.add[?] (%1142 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1144 = dfcir.sub[?] (%1143 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %865 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1145 = dfcir.shr(%1144 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1146 = dfcir.sub[?] (%1143 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1140 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1147 = dfcir.shr(%1146 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1148 = dfcir.add[?] (%1135 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %918 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1149 = dfcir.shr(%1148 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1150 = dfcir.cast[?] (%1149 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1151 = dfcir.cast[?] (%1150 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1152 = dfcir.mul[?] (%1151 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %19 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1153 = dfcir.add[?] (%1151 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %869 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1154 = dfcir.mul[?] (%1153 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1155 = dfcir.add[?] (%1154 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1156 = dfcir.sub[?] (%1155 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %870 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1157 = dfcir.shr(%1156 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1158 = dfcir.sub[?] (%1155 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1152 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1159 = dfcir.shr(%1158 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1160 = dfcir.add[?] (%157 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1134 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1161 = dfcir.sub[?] (%1160 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %922 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1162 = dfcir.shr(%1161 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1163 = dfcir.cast[?] (%1162 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1164 = dfcir.cast[?] (%1163 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1165 = dfcir.mul[?] (%1164 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1166 = dfcir.add[?] (%1164 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %878 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1167 = dfcir.mul[?] (%1166 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1168 = dfcir.add[?] (%1167 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1169 = dfcir.sub[?] (%1168 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %879 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1170 = dfcir.shr(%1169 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1171 = dfcir.sub[?] (%1168 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1165 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1172 = dfcir.shr(%1171 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1173 = dfcir.add[?] (%1160 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %922 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1174 = dfcir.shr(%1173 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1175 = dfcir.cast[?] (%1174 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1176 = dfcir.cast[?] (%1175 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1177 = dfcir.mul[?] (%1176 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %24 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1178 = dfcir.add[?] (%1176 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %883 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1179 = dfcir.mul[?] (%1178 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %133 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1180 = dfcir.add[?] (%1179 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1181 = dfcir.sub[?] (%1180 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %884 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1182 = dfcir.shr(%1181 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1183 = dfcir.sub[?] (%1180 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1177 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1184 = dfcir.shr(%1183 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1185 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %1186 = dfcir.mul[?] (%122 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1185 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1187 = dfcir.sub[?] (%134 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1186 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1188 = dfcir.sub[?] (%910 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1187 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1189 = dfcir.sub[?] (%1188 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %762 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1190 = dfcir.mul[?] (%1189 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1191 = dfcir.add[?] (%1190 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1192 = dfcir.shr(%1191 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1193 = dfcir.sub[?] (%401 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1192 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1194 = dfcir.shr(%1193 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1195 = dfcir.cast[?] (%1194 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1196 = dfcir.cast[?] (%1195 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1197 = dfcir.mul[?] (%1196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1198 = dfcir.add[?] (%545 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1199 = dfcir.mul[?] (%1198 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1200 = dfcir.add[?] (%1199 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1201 = dfcir.sub[?] (%1200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1197 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1202 = dfcir.shr(%1201 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1203 = dfcir.sub[?] (%1202 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1145 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1204 = dfcir.add[?] (%1202 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1145 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1205 = dfcir.sub[?] (%816 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1204 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1206 = dfcir.shr(%1205 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1207 = dfcir.cast[?] (%1206 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1208 = dfcir.greater[?] (%1207 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1209 = dfcir.cast[?] (%1208 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1210 = dfcir.greaterEq[?] (%1207 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1211 = dfcir.cast[?] (%1210 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1212 = dfcir.add[?] (%1211 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1209 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1213 = dfcir.mux(%1212: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1207: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1214 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out37") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1214 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1213 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1215 = dfcir.add[?] (%816 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1204 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1216 = dfcir.shr(%1215 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1217 = dfcir.cast[?] (%1216 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1218 = dfcir.greater[?] (%1217 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1219 = dfcir.cast[?] (%1218 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1220 = dfcir.greaterEq[?] (%1217 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1221 = dfcir.cast[?] (%1220 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1222 = dfcir.add[?] (%1221 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1219 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1223 = dfcir.mux(%1222: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1217: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1224 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out29") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1224 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1223 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1225 = dfcir.add[?] (%1200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %546 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1226 = dfcir.shr(%1225 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1227 = dfcir.sub[?] (%1226 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1147 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1228 = dfcir.sub[?] (%1227 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1203 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1229 = dfcir.mul[?] (%1228 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1230 = dfcir.add[?] (%1229 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1231 = dfcir.shr(%1230 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1232 = dfcir.sub[?] (%813 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1231 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1233 = dfcir.shr(%1232 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1234 = dfcir.cast[?] (%1233 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1235 = dfcir.greater[?] (%1234 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1236 = dfcir.cast[?] (%1235 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1237 = dfcir.greaterEq[?] (%1234 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1238 = dfcir.cast[?] (%1237 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1239 = dfcir.add[?] (%1238 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1236 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1240 = dfcir.mux(%1239: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1234: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1241 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out45") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1241 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1240 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1242 = dfcir.add[?] (%813 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1231 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1243 = dfcir.shr(%1242 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1244 = dfcir.cast[?] (%1243 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1245 = dfcir.greater[?] (%1244 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1246 = dfcir.cast[?] (%1245 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1247 = dfcir.greaterEq[?] (%1244 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1248 = dfcir.cast[?] (%1247 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1249 = dfcir.add[?] (%1248 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1246 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1250 = dfcir.mux(%1249: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1244: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1251 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out21") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1251 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1250 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1252 = dfcir.add[?] (%1227 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1203 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1253 = dfcir.mul[?] (%1252 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1254 = dfcir.add[?] (%1253 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1255 = dfcir.shr(%1254 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1256 = dfcir.sub[?] (%814 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1255 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1257 = dfcir.shr(%1256 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1258 = dfcir.cast[?] (%1257 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1259 = dfcir.greater[?] (%1258 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1260 = dfcir.cast[?] (%1259 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1261 = dfcir.greaterEq[?] (%1258 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1262 = dfcir.cast[?] (%1261 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1263 = dfcir.add[?] (%1262 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1260 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1264 = dfcir.mux(%1263: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1258: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1265 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out53") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1265 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1264 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1266 = dfcir.add[?] (%814 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1255 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1267 = dfcir.shr(%1266 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1268 = dfcir.cast[?] (%1267 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1269 = dfcir.greater[?] (%1268 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1270 = dfcir.cast[?] (%1269 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1271 = dfcir.greaterEq[?] (%1268 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1272 = dfcir.cast[?] (%1271 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1273 = dfcir.add[?] (%1272 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1270 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1274 = dfcir.mux(%1273: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1268: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1275 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out13") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1275 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1274 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1276 = dfcir.add[?] (%1226 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1147 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1277 = dfcir.sub[?] (%817 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1276 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1278 = dfcir.shr(%1277 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1279 = dfcir.cast[?] (%1278 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1280 = dfcir.greater[?] (%1279 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1281 = dfcir.cast[?] (%1280 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1282 = dfcir.greaterEq[?] (%1279 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1283 = dfcir.cast[?] (%1282 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1284 = dfcir.add[?] (%1283 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1281 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1285 = dfcir.mux(%1284: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1279: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1286 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out61") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1286 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1285 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1287 = dfcir.add[?] (%817 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1276 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1288 = dfcir.shr(%1287 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1289 = dfcir.cast[?] (%1288 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1290 = dfcir.greater[?] (%1289 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1291 = dfcir.cast[?] (%1290 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1292 = dfcir.greaterEq[?] (%1289 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1293 = dfcir.cast[?] (%1292 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1294 = dfcir.add[?] (%1293 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1291 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1295 = dfcir.mux(%1294: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1289: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1296 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out5") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1296 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1295 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1297 = dfcir.add[?] (%401 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1192 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1298 = dfcir.shr(%1297 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1299 = dfcir.cast[?] (%1298 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1300 = dfcir.cast[?] (%1299 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1301 = dfcir.mul[?] (%1300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %20 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1302 = dfcir.add[?] (%550 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1303 = dfcir.mul[?] (%1302 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1304 = dfcir.add[?] (%1303 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1305 = dfcir.sub[?] (%1304 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1301 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1306 = dfcir.shr(%1305 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1307 = dfcir.sub[?] (%1306 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1157 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1308 = dfcir.add[?] (%1306 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1157 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1309 = dfcir.sub[?] (%827 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1310 = dfcir.shr(%1309 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1311 = dfcir.cast[?] (%1310 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1312 = dfcir.greater[?] (%1311 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1313 = dfcir.cast[?] (%1312 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1314 = dfcir.greaterEq[?] (%1311 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1315 = dfcir.cast[?] (%1314 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1316 = dfcir.add[?] (%1315 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1313 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1317 = dfcir.mux(%1316: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1311: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1318 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out34") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1318 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1317 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1319 = dfcir.add[?] (%827 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1320 = dfcir.shr(%1319 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1321 = dfcir.cast[?] (%1320 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1322 = dfcir.greater[?] (%1321 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1323 = dfcir.cast[?] (%1322 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1324 = dfcir.greaterEq[?] (%1321 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1325 = dfcir.cast[?] (%1324 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1326 = dfcir.add[?] (%1325 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1323 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1327 = dfcir.mux(%1326: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1321: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1328 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out26") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1328 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1327 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1329 = dfcir.add[?] (%1304 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %551 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1330 = dfcir.shr(%1329 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1331 = dfcir.sub[?] (%1330 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1159 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1332 = dfcir.sub[?] (%1331 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1307 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1333 = dfcir.mul[?] (%1332 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1334 = dfcir.add[?] (%1333 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1335 = dfcir.shr(%1334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1336 = dfcir.sub[?] (%824 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1335 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1337 = dfcir.shr(%1336 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1338 = dfcir.cast[?] (%1337 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1339 = dfcir.greater[?] (%1338 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1340 = dfcir.cast[?] (%1339 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1341 = dfcir.greaterEq[?] (%1338 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1342 = dfcir.cast[?] (%1341 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1343 = dfcir.add[?] (%1342 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1340 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1344 = dfcir.mux(%1343: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1338: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1345 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out42") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1345 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1344 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1346 = dfcir.add[?] (%824 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1335 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1347 = dfcir.shr(%1346 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1348 = dfcir.cast[?] (%1347 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1349 = dfcir.greater[?] (%1348 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1350 = dfcir.cast[?] (%1349 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1351 = dfcir.greaterEq[?] (%1348 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1352 = dfcir.cast[?] (%1351 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1353 = dfcir.add[?] (%1352 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1350 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1354 = dfcir.mux(%1353: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1348: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1355 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out18") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1355 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1354 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1356 = dfcir.add[?] (%1331 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1307 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1357 = dfcir.mul[?] (%1356 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1358 = dfcir.add[?] (%1357 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1359 = dfcir.shr(%1358 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1360 = dfcir.sub[?] (%825 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1359 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1361 = dfcir.shr(%1360 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1362 = dfcir.cast[?] (%1361 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1363 = dfcir.greater[?] (%1362 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1364 = dfcir.cast[?] (%1363 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1365 = dfcir.greaterEq[?] (%1362 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1366 = dfcir.cast[?] (%1365 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1367 = dfcir.add[?] (%1366 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1364 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1368 = dfcir.mux(%1367: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1362: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1369 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out50") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1369 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1368 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1370 = dfcir.add[?] (%825 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1359 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1371 = dfcir.shr(%1370 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1372 = dfcir.cast[?] (%1371 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1373 = dfcir.greater[?] (%1372 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1374 = dfcir.cast[?] (%1373 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1375 = dfcir.greaterEq[?] (%1372 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1376 = dfcir.cast[?] (%1375 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1377 = dfcir.add[?] (%1376 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1374 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1378 = dfcir.mux(%1377: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1372: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1379 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out10") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1379 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1378 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1380 = dfcir.add[?] (%1330 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1159 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1381 = dfcir.sub[?] (%828 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1380 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1382 = dfcir.shr(%1381 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1383 = dfcir.cast[?] (%1382 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1384 = dfcir.greater[?] (%1383 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1385 = dfcir.cast[?] (%1384 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1386 = dfcir.greaterEq[?] (%1383 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1387 = dfcir.cast[?] (%1386 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1388 = dfcir.add[?] (%1387 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1385 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1389 = dfcir.mux(%1388: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1383: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1390 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out58") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1390 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1389 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1391 = dfcir.add[?] (%828 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1380 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1392 = dfcir.shr(%1391 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1393 = dfcir.cast[?] (%1392 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1394 = dfcir.greater[?] (%1393 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1395 = dfcir.cast[?] (%1394 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1396 = dfcir.greaterEq[?] (%1393 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1397 = dfcir.cast[?] (%1396 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1398 = dfcir.add[?] (%1397 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1395 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1399 = dfcir.mux(%1398: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1393: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1400 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out2") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1400 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1399 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1401 = dfcir.add[?] (%1188 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %762 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1402 = dfcir.mul[?] (%1401 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1403 = dfcir.add[?] (%1402 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1404 = dfcir.shr(%1403 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1405 = dfcir.sub[?] (%402 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1404 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1406 = dfcir.shr(%1405 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1407 = dfcir.cast[?] (%1406 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1408 = dfcir.cast[?] (%1407 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1409 = dfcir.mul[?] (%1408 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %213 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1410 = dfcir.add[?] (%559 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1408 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1411 = dfcir.mul[?] (%1410 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1412 = dfcir.add[?] (%1411 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1413 = dfcir.sub[?] (%1412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1409 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1414 = dfcir.shr(%1413 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1415 = dfcir.sub[?] (%1414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1170 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1416 = dfcir.add[?] (%1414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1170 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1417 = dfcir.sub[?] (%839 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1418 = dfcir.shr(%1417 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1419 = dfcir.cast[?] (%1418 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1420 = dfcir.greater[?] (%1419 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1421 = dfcir.cast[?] (%1420 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1422 = dfcir.greaterEq[?] (%1419 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1423 = dfcir.cast[?] (%1422 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1424 = dfcir.add[?] (%1423 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1421 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1425 = dfcir.mux(%1424: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1419: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1426 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out38") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1426 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1425 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1427 = dfcir.add[?] (%839 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1428 = dfcir.shr(%1427 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1429 = dfcir.cast[?] (%1428 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1430 = dfcir.greater[?] (%1429 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1431 = dfcir.cast[?] (%1430 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1432 = dfcir.greaterEq[?] (%1429 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1433 = dfcir.cast[?] (%1432 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1434 = dfcir.add[?] (%1433 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1431 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1435 = dfcir.mux(%1434: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1429: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1436 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out30") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1436 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1435 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1437 = dfcir.add[?] (%1412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %560 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1438 = dfcir.shr(%1437 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1439 = dfcir.sub[?] (%1438 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1172 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1440 = dfcir.sub[?] (%1439 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1415 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1441 = dfcir.mul[?] (%1440 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1442 = dfcir.add[?] (%1441 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1443 = dfcir.shr(%1442 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1444 = dfcir.sub[?] (%836 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1443 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1445 = dfcir.shr(%1444 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1446 = dfcir.cast[?] (%1445 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1447 = dfcir.greater[?] (%1446 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1448 = dfcir.cast[?] (%1447 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1449 = dfcir.greaterEq[?] (%1446 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1450 = dfcir.cast[?] (%1449 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1451 = dfcir.add[?] (%1450 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1448 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1452 = dfcir.mux(%1451: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1446: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1453 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out46") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1453 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1452 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1454 = dfcir.add[?] (%836 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1443 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1455 = dfcir.shr(%1454 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1456 = dfcir.cast[?] (%1455 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1457 = dfcir.greater[?] (%1456 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1458 = dfcir.cast[?] (%1457 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1459 = dfcir.greaterEq[?] (%1456 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1460 = dfcir.cast[?] (%1459 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1461 = dfcir.add[?] (%1460 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1458 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1462 = dfcir.mux(%1461: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1456: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1463 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out22") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1463 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1462 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1464 = dfcir.add[?] (%1439 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1415 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1465 = dfcir.mul[?] (%1464 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1466 = dfcir.add[?] (%1465 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1467 = dfcir.shr(%1466 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1468 = dfcir.sub[?] (%837 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1467 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1469 = dfcir.shr(%1468 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1470 = dfcir.cast[?] (%1469 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1471 = dfcir.greater[?] (%1470 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1472 = dfcir.cast[?] (%1471 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1473 = dfcir.greaterEq[?] (%1470 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1474 = dfcir.cast[?] (%1473 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1475 = dfcir.add[?] (%1474 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1472 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1476 = dfcir.mux(%1475: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1470: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1477 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out54") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1477 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1476 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1478 = dfcir.add[?] (%837 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1467 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1479 = dfcir.shr(%1478 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1480 = dfcir.cast[?] (%1479 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1481 = dfcir.greater[?] (%1480 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1482 = dfcir.cast[?] (%1481 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1483 = dfcir.greaterEq[?] (%1480 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1484 = dfcir.cast[?] (%1483 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1485 = dfcir.add[?] (%1484 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1482 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1486 = dfcir.mux(%1485: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1480: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1487 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out14") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1487 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1486 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1488 = dfcir.add[?] (%1438 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1172 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1489 = dfcir.sub[?] (%840 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1488 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1490 = dfcir.shr(%1489 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1491 = dfcir.cast[?] (%1490 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1492 = dfcir.greater[?] (%1491 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1493 = dfcir.cast[?] (%1492 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1494 = dfcir.greaterEq[?] (%1491 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1495 = dfcir.cast[?] (%1494 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1496 = dfcir.add[?] (%1495 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1493 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1497 = dfcir.mux(%1496: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1491: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1498 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out62") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1498 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1497 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1499 = dfcir.add[?] (%840 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1488 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1500 = dfcir.shr(%1499 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1501 = dfcir.cast[?] (%1500 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1502 = dfcir.greater[?] (%1501 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1503 = dfcir.cast[?] (%1502 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1504 = dfcir.greaterEq[?] (%1501 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1505 = dfcir.cast[?] (%1504 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1506 = dfcir.add[?] (%1505 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1503 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1507 = dfcir.mux(%1506: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1501: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1508 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out6") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1508 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1507 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1509 = dfcir.add[?] (%402 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1404 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1510 = dfcir.shr(%1509 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1511 = dfcir.cast[?] (%1510 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1512 = dfcir.cast[?] (%1511 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1513 = dfcir.mul[?] (%1512 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %579 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1514 = dfcir.add[?] (%564 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1512 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1515 = dfcir.mul[?] (%1514 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1516 = dfcir.add[?] (%1515 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1517 = dfcir.sub[?] (%1516 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1513 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1518 = dfcir.shr(%1517 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1519 = dfcir.sub[?] (%1518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1182 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1520 = dfcir.add[?] (%1518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1182 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1521 = dfcir.sub[?] (%850 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1522 = dfcir.shr(%1521 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1523 = dfcir.cast[?] (%1522 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1524 = dfcir.greater[?] (%1523 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1525 = dfcir.cast[?] (%1524 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1526 = dfcir.greaterEq[?] (%1523 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1527 = dfcir.cast[?] (%1526 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1528 = dfcir.add[?] (%1527 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1525 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1529 = dfcir.mux(%1528: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1523: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1530 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out33") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1530 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1529 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1531 = dfcir.add[?] (%850 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1532 = dfcir.shr(%1531 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1533 = dfcir.cast[?] (%1532 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1534 = dfcir.greater[?] (%1533 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1535 = dfcir.cast[?] (%1534 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1536 = dfcir.greaterEq[?] (%1533 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1537 = dfcir.cast[?] (%1536 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1538 = dfcir.add[?] (%1537 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1535 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1539 = dfcir.mux(%1538: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1533: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1540 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out25") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1540 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1539 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1541 = dfcir.add[?] (%1516 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %565 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1542 = dfcir.shr(%1541 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1543 = dfcir.sub[?] (%1542 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1544 = dfcir.sub[?] (%1543 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1519 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1545 = dfcir.mul[?] (%1544 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1546 = dfcir.add[?] (%1545 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1547 = dfcir.shr(%1546 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1548 = dfcir.sub[?] (%847 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1547 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1549 = dfcir.shr(%1548 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1550 = dfcir.cast[?] (%1549 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1551 = dfcir.greater[?] (%1550 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1552 = dfcir.cast[?] (%1551 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1553 = dfcir.greaterEq[?] (%1550 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1554 = dfcir.cast[?] (%1553 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1555 = dfcir.add[?] (%1554 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1552 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1556 = dfcir.mux(%1555: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1550: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1557 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out41") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1557 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1556 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1558 = dfcir.add[?] (%847 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1547 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1559 = dfcir.shr(%1558 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1560 = dfcir.cast[?] (%1559 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1561 = dfcir.greater[?] (%1560 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1562 = dfcir.cast[?] (%1561 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1563 = dfcir.greaterEq[?] (%1560 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1564 = dfcir.cast[?] (%1563 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1565 = dfcir.add[?] (%1564 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1562 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1566 = dfcir.mux(%1565: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1560: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1567 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out17") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1567 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1566 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1568 = dfcir.add[?] (%1543 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1519 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1569 = dfcir.mul[?] (%1568 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1570 = dfcir.add[?] (%1569 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1571 = dfcir.shr(%1570 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1572 = dfcir.sub[?] (%848 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1571 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1573 = dfcir.shr(%1572 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1574 = dfcir.cast[?] (%1573 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1575 = dfcir.greater[?] (%1574 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1576 = dfcir.cast[?] (%1575 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1577 = dfcir.greaterEq[?] (%1574 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1578 = dfcir.cast[?] (%1577 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1579 = dfcir.add[?] (%1578 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1576 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1580 = dfcir.mux(%1579: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1574: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1581 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out49") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1581 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1580 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1582 = dfcir.add[?] (%848 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1571 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1583 = dfcir.shr(%1582 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1584 = dfcir.cast[?] (%1583 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1585 = dfcir.greater[?] (%1584 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1586 = dfcir.cast[?] (%1585 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1587 = dfcir.greaterEq[?] (%1584 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1588 = dfcir.cast[?] (%1587 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1589 = dfcir.add[?] (%1588 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1586 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1590 = dfcir.mux(%1589: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1584: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1591 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out9") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1591 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1590 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1592 = dfcir.add[?] (%1542 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1593 = dfcir.sub[?] (%851 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1592 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1594 = dfcir.shr(%1593 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1595 = dfcir.cast[?] (%1594 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1596 = dfcir.greater[?] (%1595 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1597 = dfcir.cast[?] (%1596 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1598 = dfcir.greaterEq[?] (%1595 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1599 = dfcir.cast[?] (%1598 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1600 = dfcir.add[?] (%1599 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1597 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1601 = dfcir.mux(%1600: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1595: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1602 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out57") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1602 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1601 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1603 = dfcir.add[?] (%851 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1592 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1604 = dfcir.shr(%1603 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1605 = dfcir.cast[?] (%1604 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1606 = dfcir.greater[?] (%1605 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1607 = dfcir.cast[?] (%1606 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1608 = dfcir.greaterEq[?] (%1605 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1609 = dfcir.cast[?] (%1608 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1610 = dfcir.add[?] (%1609 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1607 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1611 = dfcir.mux(%1610: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1605: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1612 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out1") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1612 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1611 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1613 = dfcir.add[?] (%910 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1187 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1614 = dfcir.sub[?] (%407 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1613 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1615 = dfcir.shr(%1614 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1616 = dfcir.cast[?] (%1615 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1617 = dfcir.cast[?] (%1616 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1618 = dfcir.mul[?] (%1617 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1619 = dfcir.add[?] (%570 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1617 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1620 = dfcir.mul[?] (%1619 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1621 = dfcir.add[?] (%1620 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1622 = dfcir.sub[?] (%1621 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1618 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1623 = dfcir.shr(%1622 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1624 = dfcir.sub[?] (%1623 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %895 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1625 = dfcir.add[?] (%1623 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %895 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1626 = dfcir.sub[?] (%734 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1625 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1627 = dfcir.shr(%1626 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1628 = dfcir.cast[?] (%1627 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1629 = dfcir.greater[?] (%1628 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1630 = dfcir.cast[?] (%1629 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1631 = dfcir.greaterEq[?] (%1628 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1632 = dfcir.cast[?] (%1631 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1633 = dfcir.add[?] (%1632 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1630 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1634 = dfcir.mux(%1633: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1628: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1635 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out39") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1635 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1634 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1636 = dfcir.add[?] (%734 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1625 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1637 = dfcir.shr(%1636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1638 = dfcir.cast[?] (%1637 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1639 = dfcir.greater[?] (%1638 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1640 = dfcir.cast[?] (%1639 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1641 = dfcir.greaterEq[?] (%1638 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1642 = dfcir.cast[?] (%1641 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1643 = dfcir.add[?] (%1642 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1640 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1644 = dfcir.mux(%1643: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1638: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1645 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out31") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1645 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1644 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1646 = dfcir.add[?] (%1621 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %571 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1647 = dfcir.shr(%1646 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1648 = dfcir.sub[?] (%1647 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %897 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1649 = dfcir.sub[?] (%1648 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1624 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1650 = dfcir.mul[?] (%1649 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1651 = dfcir.add[?] (%1650 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1652 = dfcir.shr(%1651 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1653 = dfcir.sub[?] (%738 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1652 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1654 = dfcir.shr(%1653 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1655 = dfcir.cast[?] (%1654 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1656 = dfcir.greater[?] (%1655 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1657 = dfcir.cast[?] (%1656 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1658 = dfcir.greaterEq[?] (%1655 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1659 = dfcir.cast[?] (%1658 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1660 = dfcir.add[?] (%1659 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1657 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1661 = dfcir.mux(%1660: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1655: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1662 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out47") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1662 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1661 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1663 = dfcir.add[?] (%738 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1652 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1664 = dfcir.shr(%1663 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1665 = dfcir.cast[?] (%1664 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1666 = dfcir.greater[?] (%1665 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1667 = dfcir.cast[?] (%1666 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1668 = dfcir.greaterEq[?] (%1665 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1669 = dfcir.cast[?] (%1668 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1670 = dfcir.add[?] (%1669 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1667 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1671 = dfcir.mux(%1670: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1665: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1672 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out23") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1672 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1671 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1673 = dfcir.add[?] (%1648 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1624 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1674 = dfcir.mul[?] (%1673 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1675 = dfcir.add[?] (%1674 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1676 = dfcir.shr(%1675 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1677 = dfcir.sub[?] (%739 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1676 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1678 = dfcir.shr(%1677 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1679 = dfcir.cast[?] (%1678 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1680 = dfcir.greater[?] (%1679 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1681 = dfcir.cast[?] (%1680 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1682 = dfcir.greaterEq[?] (%1679 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1683 = dfcir.cast[?] (%1682 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1684 = dfcir.add[?] (%1683 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1681 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1685 = dfcir.mux(%1684: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1679: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1686 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out55") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1686 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1685 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1687 = dfcir.add[?] (%739 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1676 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1688 = dfcir.shr(%1687 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1689 = dfcir.cast[?] (%1688 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1690 = dfcir.greater[?] (%1689 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1691 = dfcir.cast[?] (%1690 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1692 = dfcir.greaterEq[?] (%1689 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1693 = dfcir.cast[?] (%1692 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1694 = dfcir.add[?] (%1693 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1691 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1695 = dfcir.mux(%1694: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1689: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1696 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out15") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1696 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1695 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1697 = dfcir.add[?] (%1647 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %897 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1698 = dfcir.sub[?] (%735 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1697 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1699 = dfcir.shr(%1698 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1700 = dfcir.cast[?] (%1699 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1701 = dfcir.greater[?] (%1700 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1702 = dfcir.cast[?] (%1701 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1703 = dfcir.greaterEq[?] (%1700 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1704 = dfcir.cast[?] (%1703 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1705 = dfcir.add[?] (%1704 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1702 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1706 = dfcir.mux(%1705: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1700: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1707 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out63") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1707 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1706 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1708 = dfcir.add[?] (%735 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1697 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1709 = dfcir.shr(%1708 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1710 = dfcir.cast[?] (%1709 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1711 = dfcir.greater[?] (%1710 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1712 = dfcir.cast[?] (%1711 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1713 = dfcir.greaterEq[?] (%1710 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1714 = dfcir.cast[?] (%1713 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1715 = dfcir.add[?] (%1714 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1712 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1716 = dfcir.mux(%1715: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1710: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1717 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out7") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1717 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1716 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1718 = dfcir.add[?] (%407 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1613 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1719 = dfcir.shr(%1718 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1720 = dfcir.cast[?] (%1719 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1721 = dfcir.cast[?] (%1720 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1722 = dfcir.mul[?] (%1721 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %110 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1723 = dfcir.add[?] (%575 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1721 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1724 = dfcir.mul[?] (%1723 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %191 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1725 = dfcir.add[?] (%1724 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1726 = dfcir.sub[?] (%1725 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1722 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1727 = dfcir.shr(%1726 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1728 = dfcir.sub[?] (%1727 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %907 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1729 = dfcir.add[?] (%1727 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %907 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1730 = dfcir.sub[?] (%750 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1729 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1731 = dfcir.shr(%1730 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1732 = dfcir.cast[?] (%1731 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1733 = dfcir.greater[?] (%1732 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1734 = dfcir.cast[?] (%1733 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1735 = dfcir.greaterEq[?] (%1732 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1736 = dfcir.cast[?] (%1735 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1737 = dfcir.add[?] (%1736 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1734 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1738 = dfcir.mux(%1737: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1732: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1739 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out32") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1739 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1738 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1740 = dfcir.add[?] (%750 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1729 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1741 = dfcir.shr(%1740 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1742 = dfcir.cast[?] (%1741 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1743 = dfcir.greater[?] (%1742 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1744 = dfcir.cast[?] (%1743 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1745 = dfcir.greaterEq[?] (%1742 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1746 = dfcir.cast[?] (%1745 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1747 = dfcir.add[?] (%1746 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1744 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1748 = dfcir.mux(%1747: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1742: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1749 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out24") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1749 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1748 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1750 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %1751 = dfcir.mul[?] (%575 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1750 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1752 = dfcir.add[?] (%1725 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1751 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1753 = dfcir.shr(%1752 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1754 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %1755 = dfcir.mul[?] (%509 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1754 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1756 = dfcir.sub[?] (%905 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1755 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1757 = dfcir.shr(%1756 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1758 = dfcir.sub[?] (%1753 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1757 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1759 = dfcir.sub[?] (%1758 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1728 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1760 = dfcir.mul[?] (%1759 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1761 = dfcir.add[?] (%1760 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1762 = dfcir.shr(%1761 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1763 = dfcir.sub[?] (%754 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1762 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1764 = dfcir.shr(%1763 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1765 = dfcir.cast[?] (%1764 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1766 = dfcir.greater[?] (%1765 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1767 = dfcir.cast[?] (%1766 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1768 = dfcir.greaterEq[?] (%1765 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1769 = dfcir.cast[?] (%1768 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1770 = dfcir.add[?] (%1769 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1767 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1771 = dfcir.mux(%1770: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1765: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1772 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out40") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1772 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1771 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1773 = dfcir.add[?] (%754 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1762 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1774 = dfcir.shr(%1773 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1775 = dfcir.cast[?] (%1774 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1776 = dfcir.greater[?] (%1775 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1777 = dfcir.cast[?] (%1776 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1778 = dfcir.greaterEq[?] (%1775 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1779 = dfcir.cast[?] (%1778 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1780 = dfcir.add[?] (%1779 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1777 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1781 = dfcir.mux(%1780: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1775: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1782 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out16") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1782 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1781 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1783 = dfcir.add[?] (%1758 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1728 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1784 = dfcir.mul[?] (%1783 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1785 = dfcir.add[?] (%1784 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1786 = dfcir.shr(%1785 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1787 = dfcir.sub[?] (%755 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1786 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1788 = dfcir.shr(%1787 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1789 = dfcir.cast[?] (%1788 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1790 = dfcir.greater[?] (%1789 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1791 = dfcir.cast[?] (%1790 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1792 = dfcir.greaterEq[?] (%1789 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1793 = dfcir.cast[?] (%1792 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1794 = dfcir.add[?] (%1793 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1791 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1795 = dfcir.mux(%1794: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1789: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1796 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out48") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1796 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1795 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1797 = dfcir.add[?] (%755 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1786 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1798 = dfcir.shr(%1797 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1799 = dfcir.cast[?] (%1798 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1800 = dfcir.greater[?] (%1799 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1801 = dfcir.cast[?] (%1800 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1802 = dfcir.greaterEq[?] (%1799 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1803 = dfcir.cast[?] (%1802 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1804 = dfcir.add[?] (%1803 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1801 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1805 = dfcir.mux(%1804: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1799: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1806 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out8") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1806 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1805 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1807 = dfcir.add[?] (%1753 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1757 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1808 = dfcir.sub[?] (%751 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1807 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1809 = dfcir.shr(%1808 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1810 = dfcir.cast[?] (%1809 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1811 = dfcir.greater[?] (%1810 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1812 = dfcir.cast[?] (%1811 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1813 = dfcir.greaterEq[?] (%1810 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1814 = dfcir.cast[?] (%1813 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1815 = dfcir.add[?] (%1814 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1812 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1816 = dfcir.mux(%1815: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1810: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1817 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out56") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1817 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1816 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
    %1818 = dfcir.add[?] (%751 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1807 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1819 = dfcir.shr(%1818 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1820 = dfcir.cast[?] (%1819 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 15, 0>> {latency = -1 : i32}
    %1821 = dfcir.greater[?] (%1820 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %100 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1822 = dfcir.cast[?] (%1821 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1823 = dfcir.greaterEq[?] (%1820 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323 : !dfcir.const<!dfcir.fixed<true, 15, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1824 = dfcir.cast[?] (%1823 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1825 = dfcir.add[?] (%1824 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1822 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1826 = dfcir.mux(%1825: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %100, %1820: !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %323) : !dfcir.const<!dfcir.fixed<true, 15, 0>>
    %1827 = dfcir.output<!dfcir.fixed<true, 15, 0>> ("out0") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1827 : !dfcir.stream<!dfcir.fixed<true, 15, 0>>, %1826 : !dfcir.const<!dfcir.fixed<true, 15, 0>>)
  }
}
