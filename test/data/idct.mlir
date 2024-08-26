module {
  dfcir.kernel "IDCT" {
    %0 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x63")
    %1 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x62")
    %2 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x61")
    %3 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x60")
    %4 = dfcir.shl(%3 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %5 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x58")
    %6 = dfcir.add(%5 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %7 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x57")
    %8 = dfcir.add(%7 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %9 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x56")
    %10 = dfcir.shl(%9 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %11 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x55")
    %12 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x54")
    %13 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x18")
    %14 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1108 : si32
    %15 = dfcir.mul(%6 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %16 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x39")
    %17 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x38")
    %18 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x53")
    %19 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x16")
    %20 = dfcir.shl(%19 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %21 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x26")
    %22 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x25")
    %23 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x24")
    %24 = dfcir.shl(%23 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %25 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x23")
    %26 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x12")
    %27 = dfcir.shl(%26 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %28 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x22")
    %29 = dfcir.add(%13 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %28 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %30 = dfcir.mul(%29 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %31 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x52")
    %32 = dfcir.shl(%31 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %33 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x15")
    %34 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x9")
    %35 = dfcir.add(%34 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %33 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %36 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x49")
    %37 = dfcir.add(%36 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %11 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %38 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x32")
    %39 = dfcir.shl(%38 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %40 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x34")
    %41 = dfcir.add(%40 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %17 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %42 = dfcir.mul(%41 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %43 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x6")
    %44 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x51")
    %45 = dfcir.add(%18 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %44 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %46 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x14")
    %47 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x4")
    %48 = dfcir.shl(%47 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %49 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x59")
    %50 = dfcir.add(%2 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %49 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %51 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x1")
    %52 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x5")
    %53 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 1609 : si32
    %54 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x7")
    %55 = dfcir.add(%51 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %54 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %56 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x11")
    %57 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x17")
    %58 = dfcir.add(%57 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %25 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %59 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2676 : si32
    %60 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %61 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %62 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %63 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %64 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %65 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %66 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %67 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %68 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %69 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %70 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %71 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %72 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %73 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %74 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %75 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %76 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %77 = dfcir.mul(%76 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %5 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %78 = dfcir.add(%15 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %77 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %79 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %80 = dfcir.mul(%79 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %81 = dfcir.sub(%15 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %80 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %82 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %83 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %84 = dfcir.mul(%83 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %12 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %85 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %86 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %87 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %88 = dfcir.mul(%87 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %40 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %89 = dfcir.add(%42 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %88 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %90 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %91 = dfcir.mul(%90 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %17 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %92 = dfcir.sub(%42 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %91 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %93 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %94 = dfcir.mul(%93 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %21 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %95 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %96 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %97 = dfcir.mul(%96 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %13 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %98 = dfcir.add(%30 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %97 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %99 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %100 = dfcir.mul(%99 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %28 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %101 = dfcir.sub(%30 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %100 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %102 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %103 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %104 = dfcir.mul(%103 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %46 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %105 = dfcir.sub(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %106 = dfcir.add(%59 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %107 = dfcir.mul(%106 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %43 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %108 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x21")
    %109 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 8192 : si32
    %110 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x37")
    %111 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 181 : si32
    %112 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x31")
    %113 = dfcir.add(%22 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %112 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %114 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x48")
    %115 = dfcir.shl(%114 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %116 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 565 : si32
    %117 = dfcir.mul(%8 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %118 = dfcir.mul(%37 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %119 = dfcir.mul(%113 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %120 = dfcir.mul(%58 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %121 = dfcir.mul(%35 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %122 = dfcir.mul(%55 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %123 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x8")
    %124 = dfcir.shl(%123 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %125 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x10")
    %126 = dfcir.mul(%102 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %125 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %127 = dfcir.add(%125 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %46 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %128 = dfcir.mul(%127 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %129 = dfcir.add(%128 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %126 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %130 = dfcir.sub(%128 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %104 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %131 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x20")
    %132 = dfcir.shl(%131 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %133 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x2")
    %134 = dfcir.mul(%105 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %133 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %135 = dfcir.add(%133 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %43 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %136 = dfcir.mul(%135 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %137 = dfcir.add(%136 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %134 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %138 = dfcir.sub(%136 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %107 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %139 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x27")
    %140 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x3")
    %141 = dfcir.add(%52 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %140 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %142 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x50")
    %143 = dfcir.mul(%82 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %142 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %144 = dfcir.add(%142 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %12 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %145 = dfcir.mul(%144 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %146 = dfcir.add(%145 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %143 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %147 = dfcir.sub(%145 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %84 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %148 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x13")
    %149 = dfcir.add(%148 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %56 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %150 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2841 : si32
    %151 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %152 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %153 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %154 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %155 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %156 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %157 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %158 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %159 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %160 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %161 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %162 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %163 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %164 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %165 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %166 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %167 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %168 = dfcir.mul(%167 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %169 = dfcir.sub(%117 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %168 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %170 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %171 = dfcir.mul(%170 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %7 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %172 = dfcir.add(%117 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %171 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %173 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %174 = dfcir.mul(%173 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %11 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %175 = dfcir.sub(%118 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %174 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %176 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %177 = dfcir.mul(%176 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %36 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %178 = dfcir.add(%118 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %177 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %179 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %180 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %181 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %182 = dfcir.mul(%181 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %16 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %183 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %184 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %185 = dfcir.mul(%184 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %112 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %186 = dfcir.sub(%119 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %185 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %187 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %188 = dfcir.mul(%187 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %22 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %189 = dfcir.add(%119 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %188 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %190 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %191 = dfcir.mul(%190 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %25 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %192 = dfcir.sub(%120 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %191 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %193 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %194 = dfcir.mul(%193 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %57 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %195 = dfcir.add(%120 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %194 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %196 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %197 = dfcir.mul(%196 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %33 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %198 = dfcir.sub(%121 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %197 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %199 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %200 = dfcir.mul(%199 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %34 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %201 = dfcir.add(%121 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %200 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %202 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %203 = dfcir.mul(%202 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %54 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %204 = dfcir.sub(%122 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %203 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %205 = dfcir.sub(%150 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %206 = dfcir.mul(%205 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %51 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %207 = dfcir.add(%122 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %206 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %208 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 128 : si32
    %209 = dfcir.add(%10 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %210 = dfcir.sub(%209 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %4 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %211 = dfcir.sub(%210 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %81 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %212 = dfcir.add(%210 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %81 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %213 = dfcir.add(%209 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %4 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %214 = dfcir.sub(%213 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %78 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %215 = dfcir.add(%213 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %78 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %216 = dfcir.add(%115 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %217 = dfcir.sub(%216 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %32 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %218 = dfcir.sub(%217 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %147 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %219 = dfcir.add(%217 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %147 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %220 = dfcir.add(%216 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %32 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %221 = dfcir.sub(%220 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %146 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %222 = dfcir.add(%220 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %146 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %223 = dfcir.add(%39 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %224 = dfcir.add(%24 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %225 = dfcir.add(%20 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %226 = dfcir.sub(%225 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %132 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %227 = dfcir.sub(%226 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %101 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %228 = dfcir.add(%226 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %101 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %229 = dfcir.add(%225 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %132 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %230 = dfcir.sub(%229 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %98 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %231 = dfcir.add(%229 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %98 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %232 = dfcir.add(%124 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %233 = dfcir.sub(%232 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %27 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %234 = dfcir.sub(%233 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %130 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %235 = dfcir.add(%233 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %130 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %236 = dfcir.add(%232 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %27 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %237 = dfcir.sub(%236 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %129 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %238 = dfcir.add(%236 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %129 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %239 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 4 : si32
    %240 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x28")
    %241 = dfcir.shl(%240 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %242 = dfcir.sub(%224 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %241 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %243 = dfcir.add(%224 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %241 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %244 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x0")
    %245 = dfcir.shl(%244 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %246 = dfcir.add(%245 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %247 = dfcir.sub(%246 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %48 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %248 = dfcir.sub(%247 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %249 = dfcir.add(%247 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %138 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %250 = dfcir.add(%246 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %48 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %251 = dfcir.sub(%250 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %137 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %252 = dfcir.add(%250 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %137 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %253 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x33")
    %254 = dfcir.mul(%183 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %253 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %255 = dfcir.add(%253 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %16 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %256 = dfcir.mul(%255 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %257 = dfcir.sub(%256 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %182 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %258 = dfcir.add(%256 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %254 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %259 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x19")
    %260 = dfcir.add(%108 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %259 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %261 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x40")
    %262 = dfcir.shl(%261 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %263 = dfcir.add(%262 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %264 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x29")
    %265 = dfcir.add(%264 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %139 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %266 = dfcir.constant<!dfcir.fixed<true, 32, 0>> 2408 : si32
    %267 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %268 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %269 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %270 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %271 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %272 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %273 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %274 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %275 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %276 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %277 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %278 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %279 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %280 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %281 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %282 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %283 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %284 = dfcir.mul(%283 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %49 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %285 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %286 = dfcir.mul(%285 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %2 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %287 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %50 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %288 = dfcir.sub(%287 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %284 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %289 = dfcir.sub(%169 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %288 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %290 = dfcir.add(%169 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %288 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %291 = dfcir.sub(%214 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %290 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %292 = dfcir.shr(%291 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %293 = dfcir.mul(%157 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %292 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %294 = dfcir.add(%214 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %290 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %295 = dfcir.shr(%294 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %296 = dfcir.mul(%159 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %295 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %297 = dfcir.sub(%287 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %286 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %298 = dfcir.sub(%172 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %297 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %299 = dfcir.sub(%298 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %289 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %300 = dfcir.mul(%299 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %301 = dfcir.add(%300 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %302 = dfcir.shr(%301 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %303 = dfcir.sub(%211 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %302 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %304 = dfcir.shr(%303 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %305 = dfcir.mul(%155 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %304 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %306 = dfcir.add(%211 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %302 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %307 = dfcir.shr(%306 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %308 = dfcir.mul(%161 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %307 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %309 = dfcir.add(%298 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %289 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %310 = dfcir.mul(%309 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %311 = dfcir.add(%310 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %312 = dfcir.shr(%311 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %313 = dfcir.sub(%212 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %312 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %314 = dfcir.shr(%313 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %315 = dfcir.mul(%153 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %314 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %316 = dfcir.add(%212 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %312 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %317 = dfcir.shr(%316 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %318 = dfcir.mul(%163 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %317 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %319 = dfcir.add(%172 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %297 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %320 = dfcir.sub(%215 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %319 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %321 = dfcir.shr(%320 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %322 = dfcir.mul(%151 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %321 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %323 = dfcir.add(%215 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %319 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %324 = dfcir.shr(%323 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %325 = dfcir.mul(%165 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %324 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %326 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %327 = dfcir.mul(%326 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %44 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %328 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %329 = dfcir.mul(%328 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %18 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %330 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %45 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %331 = dfcir.sub(%330 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %327 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %332 = dfcir.sub(%175 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %331 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %333 = dfcir.add(%175 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %331 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %334 = dfcir.sub(%221 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %333 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %335 = dfcir.shr(%334 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %336 = dfcir.mul(%67 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %335 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %337 = dfcir.add(%221 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %333 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %338 = dfcir.shr(%337 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %339 = dfcir.mul(%69 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %338 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %340 = dfcir.sub(%330 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %329 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %341 = dfcir.sub(%178 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %340 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %342 = dfcir.sub(%341 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %332 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %343 = dfcir.mul(%342 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %344 = dfcir.add(%343 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %345 = dfcir.shr(%344 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %346 = dfcir.sub(%218 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %345 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %347 = dfcir.shr(%346 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %348 = dfcir.mul(%65 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %347 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %349 = dfcir.add(%218 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %345 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %350 = dfcir.shr(%349 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %351 = dfcir.mul(%71 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %350 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %352 = dfcir.add(%341 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %332 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %353 = dfcir.mul(%352 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %354 = dfcir.add(%353 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %355 = dfcir.shr(%354 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %356 = dfcir.sub(%219 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %355 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %357 = dfcir.shr(%356 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %358 = dfcir.mul(%63 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %357 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %359 = dfcir.add(%219 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %355 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %360 = dfcir.shr(%359 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %361 = dfcir.mul(%73 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %360 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %362 = dfcir.add(%178 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %340 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %363 = dfcir.sub(%222 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %362 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %364 = dfcir.shr(%363 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %365 = dfcir.mul(%61 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %364 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %366 = dfcir.add(%222 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %362 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %367 = dfcir.shr(%366 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %368 = dfcir.mul(%75 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %367 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %369 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %370 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %371 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %372 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %373 = dfcir.mul(%372 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %110 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %374 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %375 = dfcir.mul(%374 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %139 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %376 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %377 = dfcir.mul(%376 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %264 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %378 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %265 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %379 = dfcir.sub(%378 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %375 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %380 = dfcir.sub(%186 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %379 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %381 = dfcir.add(%186 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %379 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %382 = dfcir.sub(%378 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %377 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %383 = dfcir.sub(%189 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %382 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %384 = dfcir.sub(%383 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %380 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %385 = dfcir.mul(%384 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %386 = dfcir.add(%385 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %387 = dfcir.shr(%386 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %388 = dfcir.add(%383 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %380 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %389 = dfcir.mul(%388 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %390 = dfcir.add(%389 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %391 = dfcir.shr(%390 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %392 = dfcir.add(%189 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %382 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %393 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %394 = dfcir.mul(%393 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %259 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %395 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %396 = dfcir.mul(%395 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %108 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %397 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %260 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %398 = dfcir.sub(%397 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %394 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %399 = dfcir.sub(%192 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %398 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %400 = dfcir.add(%192 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %398 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %401 = dfcir.sub(%230 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %400 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %402 = dfcir.shr(%401 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %403 = dfcir.mul(%66 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %402 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %404 = dfcir.add(%402 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %335 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %405 = dfcir.mul(%404 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %406 = dfcir.add(%405 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %407 = dfcir.add(%406 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %403 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %408 = dfcir.shr(%407 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %409 = dfcir.sub(%406 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %336 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %410 = dfcir.shr(%409 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %411 = dfcir.add(%230 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %400 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %412 = dfcir.shr(%411 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %413 = dfcir.mul(%68 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %412 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %414 = dfcir.add(%412 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %338 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %415 = dfcir.mul(%414 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %416 = dfcir.add(%415 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %417 = dfcir.add(%416 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %413 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %418 = dfcir.shr(%417 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %419 = dfcir.sub(%416 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %339 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %420 = dfcir.shr(%419 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %421 = dfcir.sub(%397 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %396 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %422 = dfcir.sub(%195 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %421 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %423 = dfcir.sub(%422 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %399 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %424 = dfcir.mul(%423 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %425 = dfcir.add(%424 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %426 = dfcir.shr(%425 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %427 = dfcir.sub(%227 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %426 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %428 = dfcir.shr(%427 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %429 = dfcir.mul(%64 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %428 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %430 = dfcir.add(%428 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %347 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %431 = dfcir.mul(%430 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %432 = dfcir.add(%431 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %433 = dfcir.add(%432 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %429 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %434 = dfcir.shr(%433 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %435 = dfcir.sub(%432 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %348 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %436 = dfcir.shr(%435 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %437 = dfcir.add(%227 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %426 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %438 = dfcir.shr(%437 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %439 = dfcir.mul(%70 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %438 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %440 = dfcir.add(%438 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %350 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %441 = dfcir.mul(%440 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %442 = dfcir.add(%441 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %443 = dfcir.add(%442 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %439 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %444 = dfcir.shr(%443 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %445 = dfcir.sub(%442 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %351 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %446 = dfcir.shr(%445 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %447 = dfcir.add(%422 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %399 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %448 = dfcir.mul(%447 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %449 = dfcir.add(%448 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %450 = dfcir.shr(%449 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %451 = dfcir.sub(%228 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %450 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %452 = dfcir.shr(%451 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %453 = dfcir.mul(%62 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %452 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %454 = dfcir.add(%452 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %357 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %455 = dfcir.mul(%454 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %456 = dfcir.add(%455 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %457 = dfcir.add(%456 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %453 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %458 = dfcir.shr(%457 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %459 = dfcir.sub(%456 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %358 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %460 = dfcir.shr(%459 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %461 = dfcir.add(%228 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %450 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %462 = dfcir.shr(%461 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %463 = dfcir.mul(%72 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %462 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %464 = dfcir.add(%462 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %360 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %465 = dfcir.mul(%464 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %466 = dfcir.add(%465 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %467 = dfcir.add(%466 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %463 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %468 = dfcir.shr(%467 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %469 = dfcir.sub(%466 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %361 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %470 = dfcir.shr(%469 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %471 = dfcir.add(%195 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %421 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %472 = dfcir.sub(%231 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %471 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %473 = dfcir.shr(%472 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %474 = dfcir.mul(%60 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %473 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %475 = dfcir.add(%473 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %364 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %476 = dfcir.mul(%475 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %477 = dfcir.add(%476 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %478 = dfcir.add(%477 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %474 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %479 = dfcir.shr(%478 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %480 = dfcir.sub(%477 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %365 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %481 = dfcir.shr(%480 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %482 = dfcir.add(%231 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %471 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %483 = dfcir.shr(%482 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %484 = dfcir.mul(%74 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %483 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %485 = dfcir.add(%483 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %367 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %486 = dfcir.mul(%485 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %487 = dfcir.add(%486 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %488 = dfcir.add(%487 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %484 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %489 = dfcir.shr(%488 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %490 = dfcir.sub(%487 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %368 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %491 = dfcir.shr(%490 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %492 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %493 = dfcir.mul(%492 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %56 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %494 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %495 = dfcir.mul(%494 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %148 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %496 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %149 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %497 = dfcir.sub(%496 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %493 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %498 = dfcir.sub(%198 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %497 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %499 = dfcir.add(%198 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %497 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %500 = dfcir.sub(%237 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %499 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %501 = dfcir.shr(%500 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %502 = dfcir.mul(%158 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %501 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %503 = dfcir.add(%501 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %292 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %504 = dfcir.mul(%503 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %505 = dfcir.add(%504 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %506 = dfcir.sub(%505 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %293 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %507 = dfcir.shr(%506 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %508 = dfcir.add(%505 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %502 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %509 = dfcir.shr(%508 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %510 = dfcir.add(%237 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %499 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %511 = dfcir.shr(%510 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %512 = dfcir.mul(%160 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %511 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %513 = dfcir.add(%511 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %295 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %514 = dfcir.mul(%513 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %515 = dfcir.add(%514 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %516 = dfcir.sub(%515 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %296 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %517 = dfcir.shr(%516 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %518 = dfcir.add(%515 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %512 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %519 = dfcir.shr(%518 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %520 = dfcir.sub(%496 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %495 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %521 = dfcir.sub(%201 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %520 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %522 = dfcir.sub(%521 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %498 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %523 = dfcir.mul(%522 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %524 = dfcir.add(%523 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %525 = dfcir.shr(%524 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %526 = dfcir.sub(%234 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %525 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %527 = dfcir.shr(%526 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %528 = dfcir.mul(%156 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %527 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %529 = dfcir.add(%527 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %304 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %530 = dfcir.mul(%529 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %531 = dfcir.add(%530 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %532 = dfcir.sub(%531 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %305 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %533 = dfcir.shr(%532 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %534 = dfcir.add(%531 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %528 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %535 = dfcir.shr(%534 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %536 = dfcir.add(%234 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %525 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %537 = dfcir.shr(%536 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %538 = dfcir.mul(%162 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %537 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %539 = dfcir.add(%537 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %307 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %540 = dfcir.mul(%539 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %541 = dfcir.add(%540 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %542 = dfcir.sub(%541 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %308 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %543 = dfcir.shr(%542 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %544 = dfcir.add(%541 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %538 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %545 = dfcir.shr(%544 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %546 = dfcir.add(%521 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %498 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %547 = dfcir.mul(%546 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %548 = dfcir.add(%547 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %549 = dfcir.shr(%548 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %550 = dfcir.sub(%235 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %549 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %551 = dfcir.shr(%550 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %552 = dfcir.mul(%154 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %551 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %553 = dfcir.add(%551 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %314 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %554 = dfcir.mul(%553 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %555 = dfcir.add(%554 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %556 = dfcir.sub(%555 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %315 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %557 = dfcir.shr(%556 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %558 = dfcir.add(%555 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %552 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %559 = dfcir.shr(%558 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %560 = dfcir.add(%235 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %549 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %561 = dfcir.shr(%560 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %562 = dfcir.mul(%164 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %561 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %563 = dfcir.add(%561 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %317 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %564 = dfcir.mul(%563 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %565 = dfcir.add(%564 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %566 = dfcir.sub(%565 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %318 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %567 = dfcir.shr(%566 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %568 = dfcir.add(%565 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %562 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %569 = dfcir.shr(%568 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %570 = dfcir.add(%201 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %520 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %571 = dfcir.sub(%238 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %570 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %572 = dfcir.shr(%571 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %573 = dfcir.mul(%152 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %572 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %574 = dfcir.add(%572 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %321 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %575 = dfcir.mul(%574 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %576 = dfcir.add(%575 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %577 = dfcir.sub(%576 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %322 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %578 = dfcir.shr(%577 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %579 = dfcir.add(%576 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %573 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %580 = dfcir.shr(%579 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %581 = dfcir.add(%238 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %570 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %582 = dfcir.shr(%581 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %583 = dfcir.mul(%166 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %582 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %584 = dfcir.add(%582 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %324 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %585 = dfcir.mul(%584 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %586 = dfcir.add(%585 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %587 = dfcir.sub(%586 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %325 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %588 = dfcir.shr(%587 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %589 = dfcir.add(%586 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %583 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %590 = dfcir.shr(%589 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %591 = dfcir.add(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %592 = dfcir.mul(%591 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %140 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %593 = dfcir.sub(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %53 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.const<!dfcir.fixed<true, 32, 0>>
    %594 = dfcir.mul(%593 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %52 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %595 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %141 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %596 = dfcir.sub(%595 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %592 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %597 = dfcir.sub(%204 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %596 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %598 = dfcir.add(%204 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %596 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %599 = dfcir.sub(%251 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %598 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %600 = dfcir.shr(%599 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %601 = dfcir.shl(%600 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %602 = dfcir.add(%601 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %603 = dfcir.add(%251 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %598 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %604 = dfcir.shr(%603 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %605 = dfcir.shl(%604 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %606 = dfcir.add(%605 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %607 = dfcir.sub(%595 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %594 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %608 = dfcir.sub(%207 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %607 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %609 = dfcir.sub(%608 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %597 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %610 = dfcir.mul(%609 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %611 = dfcir.add(%610 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %612 = dfcir.shr(%611 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %613 = dfcir.sub(%248 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %612 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %614 = dfcir.shr(%613 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %615 = dfcir.shl(%614 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %616 = dfcir.add(%615 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %617 = dfcir.add(%248 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %612 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %618 = dfcir.shr(%617 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %619 = dfcir.shl(%618 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %620 = dfcir.add(%619 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %621 = dfcir.add(%608 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %597 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %622 = dfcir.mul(%621 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %623 = dfcir.add(%622 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %624 = dfcir.shr(%623 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %625 = dfcir.sub(%249 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %624 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %626 = dfcir.shr(%625 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %627 = dfcir.shl(%626 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %628 = dfcir.add(%627 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %629 = dfcir.add(%249 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %624 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %630 = dfcir.shr(%629 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %631 = dfcir.shl(%630 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %632 = dfcir.add(%631 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %633 = dfcir.add(%207 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %607 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %634 = dfcir.sub(%252 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %633 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %635 = dfcir.shr(%634 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %636 = dfcir.shl(%635 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %637 = dfcir.add(%636 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %638 = dfcir.add(%252 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %633 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %639 = dfcir.shr(%638 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %640 = dfcir.shl(%639 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %641 = dfcir.add(%640 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %109 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %642 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x35")
    %643 = dfcir.mul(%371 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %642 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %644 = dfcir.add(%110 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %642 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %645 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %644 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %646 = dfcir.sub(%645 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %643 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %647 = dfcir.sub(%257 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %646 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %648 = dfcir.add(%257 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %646 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %649 = dfcir.sub(%645 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %373 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %650 = dfcir.sub(%258 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %649 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %651 = dfcir.sub(%650 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %647 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %652 = dfcir.mul(%651 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %653 = dfcir.add(%652 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %654 = dfcir.shr(%653 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %655 = dfcir.add(%650 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %647 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %656 = dfcir.mul(%655 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %657 = dfcir.add(%656 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %658 = dfcir.shr(%657 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %659 = dfcir.add(%258 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %649 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %660 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x36")
    %661 = dfcir.shl(%660 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %662 = dfcir.sub(%223 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %661 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %663 = dfcir.sub(%662 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %92 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %664 = dfcir.sub(%663 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %654 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %665 = dfcir.shr(%664 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %666 = dfcir.shl(%665 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %667 = dfcir.sub(%616 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %666 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %668 = dfcir.sub(%667 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %436 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %669 = dfcir.add(%667 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %436 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %670 = dfcir.add(%616 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %666 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %671 = dfcir.sub(%670 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %434 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %672 = dfcir.add(%670 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %434 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %673 = dfcir.add(%663 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %654 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %674 = dfcir.shr(%673 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %675 = dfcir.shl(%674 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %676 = dfcir.sub(%620 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %675 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %677 = dfcir.sub(%676 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %446 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %678 = dfcir.add(%676 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %446 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %679 = dfcir.add(%620 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %675 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %680 = dfcir.sub(%679 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %444 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %681 = dfcir.add(%679 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %444 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %682 = dfcir.add(%662 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %92 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %683 = dfcir.sub(%682 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %658 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %684 = dfcir.shr(%683 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %685 = dfcir.shl(%684 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %686 = dfcir.sub(%628 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %685 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %687 = dfcir.sub(%686 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %460 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %688 = dfcir.add(%686 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %460 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %689 = dfcir.add(%628 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %685 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %690 = dfcir.sub(%689 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %458 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %691 = dfcir.add(%689 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %458 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %692 = dfcir.add(%682 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %658 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %693 = dfcir.shr(%692 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %694 = dfcir.shl(%693 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %695 = dfcir.sub(%632 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %694 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %696 = dfcir.sub(%695 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %470 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %697 = dfcir.add(%695 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %470 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %698 = dfcir.add(%632 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %694 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %699 = dfcir.sub(%698 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %468 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %700 = dfcir.add(%698 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %468 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %701 = dfcir.add(%223 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %661 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %702 = dfcir.sub(%701 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %89 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %703 = dfcir.sub(%702 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %648 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %704 = dfcir.shr(%703 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %705 = dfcir.shl(%704 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %706 = dfcir.sub(%602 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %705 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %707 = dfcir.sub(%706 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %410 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %708 = dfcir.add(%706 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %410 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %709 = dfcir.add(%602 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %705 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %710 = dfcir.sub(%709 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %408 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %711 = dfcir.add(%709 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %408 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %712 = dfcir.add(%702 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %648 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %713 = dfcir.shr(%712 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %714 = dfcir.shl(%713 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %715 = dfcir.sub(%606 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %714 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %716 = dfcir.sub(%715 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %420 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %717 = dfcir.add(%715 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %420 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %718 = dfcir.add(%606 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %714 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %719 = dfcir.sub(%718 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %418 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %720 = dfcir.add(%718 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %418 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %721 = dfcir.add(%701 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %89 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %722 = dfcir.sub(%721 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %659 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %723 = dfcir.shr(%722 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %724 = dfcir.shl(%723 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %725 = dfcir.sub(%637 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %724 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %726 = dfcir.sub(%725 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %481 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %727 = dfcir.add(%725 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %481 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %728 = dfcir.add(%637 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %724 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %729 = dfcir.sub(%728 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %479 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %730 = dfcir.add(%728 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %479 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %731 = dfcir.add(%721 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %659 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %732 = dfcir.shr(%731 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %733 = dfcir.shl(%732 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %734 = dfcir.sub(%641 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %733 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %735 = dfcir.sub(%734 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %491 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %736 = dfcir.add(%734 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %491 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %737 = dfcir.add(%641 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %733 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %738 = dfcir.sub(%737 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %489 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %739 = dfcir.add(%737 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %489 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %740 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x41")
    %741 = dfcir.mul(%180 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %740 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %742 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x42")
    %743 = dfcir.mul(%85 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %742 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %744 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x43")
    %745 = dfcir.mul(%369 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %744 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %746 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x44")
    %747 = dfcir.shl(%746 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %748 = dfcir.sub(%263 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %747 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %749 = dfcir.add(%263 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %747 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %750 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x45")
    %751 = dfcir.mul(%370 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %750 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %752 = dfcir.add(%750 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %744 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %753 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %752 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %754 = dfcir.sub(%753 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %745 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %755 = dfcir.sub(%753 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %751 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %756 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x46")
    %757 = dfcir.mul(%86 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %756 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %758 = dfcir.add(%742 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %756 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %759 = dfcir.mul(%758 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %760 = dfcir.add(%759 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %743 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %761 = dfcir.sub(%749 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %760 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %762 = dfcir.add(%749 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %760 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %763 = dfcir.sub(%759 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %757 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %764 = dfcir.sub(%748 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %763 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %765 = dfcir.add(%748 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %763 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %766 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x30")
    %767 = dfcir.mul(%95 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %766 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %768 = dfcir.add(%21 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %766 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %769 = dfcir.mul(%768 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %770 = dfcir.add(%769 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %94 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %771 = dfcir.sub(%243 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %770 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %772 = dfcir.sub(%771 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %381 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %773 = dfcir.shr(%772 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %774 = dfcir.mul(%273 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %773 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %775 = dfcir.add(%771 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %381 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %776 = dfcir.shr(%775 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %777 = dfcir.mul(%275 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %776 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %778 = dfcir.add(%243 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %770 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %779 = dfcir.sub(%778 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %392 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %780 = dfcir.shr(%779 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %781 = dfcir.mul(%267 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %780 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %782 = dfcir.add(%778 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %392 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %783 = dfcir.shr(%782 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %784 = dfcir.mul(%281 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %783 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %785 = dfcir.sub(%769 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %767 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %786 = dfcir.sub(%242 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %785 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %787 = dfcir.sub(%786 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %387 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %788 = dfcir.shr(%787 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %789 = dfcir.mul(%271 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %788 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %790 = dfcir.add(%786 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %387 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %791 = dfcir.shr(%790 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %792 = dfcir.mul(%277 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %791 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %793 = dfcir.add(%242 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %785 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %794 = dfcir.sub(%793 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %391 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %795 = dfcir.shr(%794 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %796 = dfcir.mul(%269 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %795 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %797 = dfcir.add(%793 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %391 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %798 = dfcir.shr(%797 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %799 = dfcir.mul(%279 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %798 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %800 = dfcir.input<!dfcir.fixed<true, 32, 0>> ("x47")
    %801 = dfcir.mul(%179 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %800 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %802 = dfcir.add(%740 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %800 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %803 = dfcir.mul(%802 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %116 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %804 = dfcir.sub(%803 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %801 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %805 = dfcir.sub(%804 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %754 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %806 = dfcir.add(%804 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %754 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %807 = dfcir.sub(%761 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %806 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %808 = dfcir.shr(%807 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %809 = dfcir.mul(%274 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %808 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %810 = dfcir.add(%808 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %773 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %811 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %810 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %812 = dfcir.add(%811 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %813 = dfcir.sub(%812 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %774 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %814 = dfcir.shr(%813 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %815 = dfcir.sub(%507 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %814 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %816 = dfcir.add(%507 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %814 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %817 = dfcir.sub(%710 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %816 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %818 = dfcir.shr(%817 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %819 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out36") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%819 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %818 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %820 = dfcir.add(%710 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %816 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %821 = dfcir.shr(%820 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %822 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out28") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%822 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %821 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %823 = dfcir.sub(%812 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %809 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %824 = dfcir.shr(%823 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %825 = dfcir.sub(%509 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %824 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %826 = dfcir.sub(%825 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %815 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %827 = dfcir.mul(%826 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %828 = dfcir.add(%827 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %829 = dfcir.shr(%828 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %830 = dfcir.sub(%707 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %829 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %831 = dfcir.shr(%830 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %832 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out44") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%832 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %831 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %833 = dfcir.add(%707 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %829 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %834 = dfcir.shr(%833 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %835 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out20") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%835 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %834 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %836 = dfcir.add(%825 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %815 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %837 = dfcir.mul(%836 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %838 = dfcir.add(%837 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %839 = dfcir.shr(%838 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %840 = dfcir.sub(%708 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %839 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %841 = dfcir.shr(%840 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %842 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out52") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%842 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %841 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %843 = dfcir.add(%708 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %839 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %844 = dfcir.shr(%843 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %845 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out12") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%845 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %844 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %846 = dfcir.add(%509 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %824 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %847 = dfcir.sub(%711 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %846 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %848 = dfcir.shr(%847 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %849 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out60") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%849 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %848 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %850 = dfcir.add(%711 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %846 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %851 = dfcir.shr(%850 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %852 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out4") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%852 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %851 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %853 = dfcir.add(%761 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %806 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %854 = dfcir.shr(%853 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %855 = dfcir.mul(%276 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %854 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %856 = dfcir.add(%854 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %776 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %857 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %856 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %858 = dfcir.add(%857 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %859 = dfcir.sub(%858 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %777 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %860 = dfcir.shr(%859 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %861 = dfcir.sub(%517 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %860 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %862 = dfcir.add(%517 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %860 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %863 = dfcir.sub(%719 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %862 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %864 = dfcir.shr(%863 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %865 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out35") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%865 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %864 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %866 = dfcir.add(%719 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %862 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %867 = dfcir.shr(%866 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %868 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out27") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%868 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %867 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %869 = dfcir.sub(%858 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %855 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %870 = dfcir.shr(%869 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %871 = dfcir.sub(%519 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %870 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %872 = dfcir.sub(%871 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %861 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %873 = dfcir.mul(%872 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %874 = dfcir.add(%873 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %875 = dfcir.shr(%874 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %876 = dfcir.sub(%716 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %875 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %877 = dfcir.shr(%876 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %878 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out43") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%878 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %877 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %879 = dfcir.add(%716 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %875 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %880 = dfcir.shr(%879 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %881 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out19") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%881 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %880 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %882 = dfcir.add(%871 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %861 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %883 = dfcir.mul(%882 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %884 = dfcir.add(%883 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %885 = dfcir.shr(%884 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %886 = dfcir.sub(%717 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %885 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %887 = dfcir.shr(%886 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %888 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out51") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%888 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %887 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %889 = dfcir.add(%717 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %885 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %890 = dfcir.shr(%889 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %891 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out11") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%891 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %890 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %892 = dfcir.add(%519 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %870 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %893 = dfcir.sub(%720 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %892 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %894 = dfcir.shr(%893 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %895 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out59") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%895 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %894 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %896 = dfcir.add(%720 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %892 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %897 = dfcir.shr(%896 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %898 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out3") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%898 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %897 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %899 = dfcir.add(%803 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %741 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %900 = dfcir.sub(%899 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %755 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %901 = dfcir.sub(%900 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %805 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %902 = dfcir.mul(%901 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %903 = dfcir.add(%902 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %904 = dfcir.shr(%903 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %905 = dfcir.sub(%764 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %904 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %906 = dfcir.shr(%905 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %907 = dfcir.mul(%272 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %906 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %908 = dfcir.add(%906 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %788 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %909 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %908 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %910 = dfcir.add(%909 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %911 = dfcir.sub(%910 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %789 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %912 = dfcir.shr(%911 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %913 = dfcir.sub(%533 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %912 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %914 = dfcir.add(%533 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %912 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %915 = dfcir.sub(%671 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %914 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %916 = dfcir.shr(%915 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %917 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out37") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%917 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %916 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %918 = dfcir.add(%671 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %914 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %919 = dfcir.shr(%918 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %920 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out29") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%920 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %919 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %921 = dfcir.sub(%910 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %907 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %922 = dfcir.shr(%921 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %923 = dfcir.sub(%535 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %922 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %924 = dfcir.sub(%923 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %913 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %925 = dfcir.mul(%924 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %926 = dfcir.add(%925 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %927 = dfcir.shr(%926 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %928 = dfcir.sub(%668 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %927 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %929 = dfcir.shr(%928 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %930 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out45") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%930 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %929 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %931 = dfcir.add(%668 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %927 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %932 = dfcir.shr(%931 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %933 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out21") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%933 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %932 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %934 = dfcir.add(%923 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %913 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %935 = dfcir.mul(%934 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %936 = dfcir.add(%935 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %937 = dfcir.shr(%936 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %938 = dfcir.sub(%669 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %937 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %939 = dfcir.shr(%938 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %940 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out53") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%940 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %939 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %941 = dfcir.add(%669 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %937 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %942 = dfcir.shr(%941 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %943 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out13") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%943 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %942 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %944 = dfcir.add(%535 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %922 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %945 = dfcir.sub(%672 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %944 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %946 = dfcir.shr(%945 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %947 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out61") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%947 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %946 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %948 = dfcir.add(%672 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %944 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %949 = dfcir.shr(%948 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %950 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out5") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%950 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %949 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %951 = dfcir.add(%764 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %904 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %952 = dfcir.shr(%951 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %953 = dfcir.mul(%278 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %952 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %954 = dfcir.add(%952 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %791 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %955 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %954 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %956 = dfcir.add(%955 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %957 = dfcir.sub(%956 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %792 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %958 = dfcir.shr(%957 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %959 = dfcir.sub(%543 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %958 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %960 = dfcir.add(%543 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %958 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %961 = dfcir.sub(%680 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %960 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %962 = dfcir.shr(%961 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %963 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out34") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%963 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %962 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %964 = dfcir.add(%680 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %960 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %965 = dfcir.shr(%964 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %966 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out26") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%966 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %965 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %967 = dfcir.sub(%956 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %953 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %968 = dfcir.shr(%967 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %969 = dfcir.sub(%545 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %968 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %970 = dfcir.sub(%969 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %959 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %971 = dfcir.mul(%970 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %972 = dfcir.add(%971 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %973 = dfcir.shr(%972 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %974 = dfcir.sub(%677 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %973 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %975 = dfcir.shr(%974 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %976 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out42") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%976 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %975 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %977 = dfcir.add(%677 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %973 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %978 = dfcir.shr(%977 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %979 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out18") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%979 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %978 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %980 = dfcir.add(%969 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %959 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %981 = dfcir.mul(%980 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %982 = dfcir.add(%981 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %983 = dfcir.shr(%982 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %984 = dfcir.sub(%678 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %983 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %985 = dfcir.shr(%984 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %986 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out50") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%986 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %985 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %987 = dfcir.add(%678 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %983 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %988 = dfcir.shr(%987 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %989 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out10") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%989 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %988 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %990 = dfcir.add(%545 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %968 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %991 = dfcir.sub(%681 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %990 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %992 = dfcir.shr(%991 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %993 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out58") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%993 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %992 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %994 = dfcir.add(%681 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %990 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %995 = dfcir.shr(%994 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %996 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out2") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%996 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %995 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %997 = dfcir.add(%900 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %805 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %998 = dfcir.mul(%997 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %999 = dfcir.add(%998 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1000 = dfcir.shr(%999 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1001 = dfcir.sub(%765 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1000 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1002 = dfcir.shr(%1001 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1003 = dfcir.mul(%270 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1002 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1004 = dfcir.add(%1002 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %795 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1005 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1004 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1006 = dfcir.add(%1005 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1007 = dfcir.sub(%1006 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %796 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1008 = dfcir.shr(%1007 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1009 = dfcir.sub(%557 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1008 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1010 = dfcir.add(%557 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1008 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1011 = dfcir.sub(%690 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1010 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1012 = dfcir.shr(%1011 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1013 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out38") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1013 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1012 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1014 = dfcir.add(%690 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1010 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1015 = dfcir.shr(%1014 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1016 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out30") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1016 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1015 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1017 = dfcir.sub(%1006 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1003 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1018 = dfcir.shr(%1017 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1019 = dfcir.sub(%559 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1018 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1020 = dfcir.sub(%1019 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1009 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1021 = dfcir.mul(%1020 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1022 = dfcir.add(%1021 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1023 = dfcir.shr(%1022 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1024 = dfcir.sub(%687 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1023 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1025 = dfcir.shr(%1024 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1026 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out46") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1026 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1025 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1027 = dfcir.add(%687 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1023 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1028 = dfcir.shr(%1027 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1029 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out22") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1029 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1028 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1030 = dfcir.add(%1019 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1009 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1031 = dfcir.mul(%1030 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1032 = dfcir.add(%1031 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1033 = dfcir.shr(%1032 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1034 = dfcir.sub(%688 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1033 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1035 = dfcir.shr(%1034 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1036 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out54") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1036 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1035 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1037 = dfcir.add(%688 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1033 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1038 = dfcir.shr(%1037 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1039 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out14") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1039 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1038 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1040 = dfcir.add(%559 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1018 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1041 = dfcir.sub(%691 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1040 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1042 = dfcir.shr(%1041 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1043 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out62") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1043 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1042 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1044 = dfcir.add(%691 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1040 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1045 = dfcir.shr(%1044 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1046 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out6") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1046 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1045 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1047 = dfcir.add(%765 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1000 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1048 = dfcir.shr(%1047 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1049 = dfcir.mul(%280 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1048 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1050 = dfcir.add(%1048 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %798 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1051 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1050 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1052 = dfcir.add(%1051 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1053 = dfcir.sub(%1052 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %799 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1054 = dfcir.shr(%1053 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1055 = dfcir.sub(%567 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1054 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1056 = dfcir.add(%567 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1054 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1057 = dfcir.sub(%699 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1056 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1058 = dfcir.shr(%1057 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1059 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out33") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1059 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1058 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1060 = dfcir.add(%699 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1056 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1061 = dfcir.shr(%1060 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1062 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out25") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1062 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1061 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1063 = dfcir.sub(%1052 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1049 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1064 = dfcir.shr(%1063 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1065 = dfcir.sub(%569 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1064 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1066 = dfcir.sub(%1065 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1055 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1067 = dfcir.mul(%1066 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1068 = dfcir.add(%1067 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1069 = dfcir.shr(%1068 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1070 = dfcir.sub(%696 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1069 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1071 = dfcir.shr(%1070 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1072 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out41") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1072 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1071 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1073 = dfcir.add(%696 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1069 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1074 = dfcir.shr(%1073 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1075 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out17") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1075 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1074 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1076 = dfcir.add(%1065 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1055 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1077 = dfcir.mul(%1076 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1078 = dfcir.add(%1077 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1079 = dfcir.shr(%1078 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1080 = dfcir.sub(%697 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1079 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1081 = dfcir.shr(%1080 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1082 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out49") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1082 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1081 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1083 = dfcir.add(%697 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1079 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1084 = dfcir.shr(%1083 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1085 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out9") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1085 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1084 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1086 = dfcir.add(%569 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1064 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1087 = dfcir.sub(%700 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1086 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1088 = dfcir.shr(%1087 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1089 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out57") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1089 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1088 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1090 = dfcir.add(%700 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1086 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1091 = dfcir.shr(%1090 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1092 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out1") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1092 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1091 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1093 = dfcir.add(%899 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %755 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1094 = dfcir.sub(%762 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1093 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1095 = dfcir.shr(%1094 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1096 = dfcir.mul(%268 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1095 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1097 = dfcir.add(%1095 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %780 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1098 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1097 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1099 = dfcir.add(%1098 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1100 = dfcir.sub(%1099 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %781 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1101 = dfcir.shr(%1100 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1102 = dfcir.sub(%578 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1101 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1103 = dfcir.add(%578 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1101 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1104 = dfcir.sub(%729 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1103 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1105 = dfcir.shr(%1104 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1106 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out39") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1106 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1105 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1107 = dfcir.add(%729 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1103 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1108 = dfcir.shr(%1107 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1109 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out31") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1109 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1108 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1110 = dfcir.sub(%1099 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1096 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1111 = dfcir.shr(%1110 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1112 = dfcir.sub(%580 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1111 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1113 = dfcir.sub(%1112 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1102 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1114 = dfcir.mul(%1113 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1115 = dfcir.add(%1114 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1116 = dfcir.shr(%1115 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1117 = dfcir.sub(%726 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1116 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1118 = dfcir.shr(%1117 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1119 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out47") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1119 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1118 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1120 = dfcir.add(%726 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1116 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1121 = dfcir.shr(%1120 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1122 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out23") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1122 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1121 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1123 = dfcir.add(%1112 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1102 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1124 = dfcir.mul(%1123 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1125 = dfcir.add(%1124 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1126 = dfcir.shr(%1125 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1127 = dfcir.sub(%727 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1126 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1128 = dfcir.shr(%1127 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1129 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out55") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1129 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1128 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1130 = dfcir.add(%727 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1126 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1131 = dfcir.shr(%1130 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1132 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out15") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1132 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1131 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1133 = dfcir.add(%580 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1111 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1134 = dfcir.sub(%730 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1133 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1135 = dfcir.shr(%1134 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1136 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out63") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1136 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1135 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1137 = dfcir.add(%730 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1133 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1138 = dfcir.shr(%1137 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1139 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out7") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1139 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1138 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1140 = dfcir.add(%762 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1093 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1141 = dfcir.shr(%1140 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1142 = dfcir.mul(%282 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1141 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1143 = dfcir.add(%1141 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>, %783 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 35, 0>>
    %1144 = dfcir.mul(%266 : !dfcir.const<!dfcir.fixed<true, 32, 0>>, %1143 : !dfcir.stream<!dfcir.fixed<true, 35, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1145 = dfcir.add(%1144 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %239 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1146 = dfcir.sub(%1145 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %784 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1147 = dfcir.shr(%1146 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1148 = dfcir.sub(%588 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1147 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1149 = dfcir.add(%588 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1147 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1150 = dfcir.sub(%738 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1149 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1151 = dfcir.shr(%1150 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1152 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out32") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1152 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1151 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1153 = dfcir.add(%738 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1149 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1154 = dfcir.shr(%1153 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1155 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out24") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1155 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1154 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1156 = dfcir.sub(%1145 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1142 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1157 = dfcir.shr(%1156 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1158 = dfcir.sub(%590 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1157 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1159 = dfcir.sub(%1158 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1148 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1160 = dfcir.mul(%1159 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1161 = dfcir.add(%1160 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1162 = dfcir.shr(%1161 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1163 = dfcir.sub(%735 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1162 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1164 = dfcir.shr(%1163 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1165 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out40") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1165 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1164 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1166 = dfcir.add(%735 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1162 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1167 = dfcir.shr(%1166 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1168 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out16") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1168 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1167 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1169 = dfcir.add(%1158 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1148 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1170 = dfcir.mul(%1169 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %111 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1171 = dfcir.add(%1170 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %208 : !dfcir.const<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1172 = dfcir.shr(%1171 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 24, 0>>
    %1173 = dfcir.sub(%736 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1172 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1174 = dfcir.shr(%1173 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1175 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out48") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1175 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1174 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1176 = dfcir.add(%736 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1172 : !dfcir.stream<!dfcir.fixed<true, 24, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1177 = dfcir.shr(%1176 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1178 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out8") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1178 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1177 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1179 = dfcir.add(%590 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1157 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>) : !dfcir.stream<!dfcir.fixed<true, 32, 0>>
    %1180 = dfcir.sub(%739 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1179 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1181 = dfcir.shr(%1180 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1182 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out56") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1182 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1181 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
    %1183 = dfcir.add(%739 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, %1179 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>) : !dfcir.stream<!dfcir.fixed<true, 43, 0>>
    %1184 = dfcir.shr(%1183 : !dfcir.stream<!dfcir.fixed<true, 43, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 29, 0>>
    %1185 = dfcir.output<!dfcir.fixed<true, 32, 0>> ("out0") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1185 : !dfcir.stream<!dfcir.fixed<true, 32, 0>>, %1184 : !dfcir.stream<!dfcir.fixed<true, 29, 0>>)
  }
}
