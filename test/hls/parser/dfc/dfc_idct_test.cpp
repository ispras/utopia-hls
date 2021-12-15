//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"

#include "gtest/gtest.h"

#include <array>
#include <fstream>
#include <iostream>

DFC_KERNEL(IDCT) {
  static dfc::value<dfc::sint32> c4;
  static dfc::value<dfc::sint32> c128;
  static dfc::value<dfc::sint32> c181;
  static dfc::value<dfc::sint32> c8192;

  static dfc::value<dfc::sint32> W1; // 2048*sqrt(2)*cos(1*pi/16)
  static dfc::value<dfc::sint32> W2; // 2048*sqrt(2)*cos(2*pi/16)
  static dfc::value<dfc::sint32> W3; // 2048*sqrt(2)*cos(3*pi/16)
  static dfc::value<dfc::sint32> W5; // 2048*sqrt(2)*cos(5*pi/16)
  static dfc::value<dfc::sint32> W6; // 2048*sqrt(2)*cos(6*pi/16)
  static dfc::value<dfc::sint32> W7; // 2048*sqrt(2)*cos(7*pi/16)

  DFC_KERNEL_CTOR(IDCT) {
    std::array<dfc::stream<dfc::sint16>, 64> blk;

    for (std::size_t i = 0; i < 8; i++)
      idctrow(blk, i);

    for (std::size_t i = 0; i < 8; i++)
      idctcol(blk, i);
  }

  /* row (horizontal) IDCT
   *
   *           7                       pi         1
   * dst[k] = sum c[l] * src[l] * cos( -- * ( k + - ) * l )
   *          l=0                      8          2
   *
   * where: c[0]    = 128
   *        c[1..7] = 128*sqrt(2)
   */
  void idctrow(std::array<dfc::stream<dfc::sint16>, 64> &blk, std::size_t i) {
    dfc::stream<dfc::sint32> x0, x1, x2, x3, x4, x5, x6, x7, x8;

    /* TODO: shortcut */
    x1 = blk[8*i+4].cast<dfc::sint32>()<<11;
    x2 = blk[8*i+6].cast<dfc::sint32>();
    x3 = blk[8*i+2].cast<dfc::sint32>();
    x4 = blk[8*i+1].cast<dfc::sint32>();
    x5 = blk[8*i+7].cast<dfc::sint32>();
    x6 = blk[8*i+5].cast<dfc::sint32>();
    x7 = blk[8*i+3].cast<dfc::sint32>();

    x0 = (blk[0].cast<dfc::sint32>()<<11) + c128; /* for proper rounding in the fourth stage */

    /* first stage */
    x8 = W7*(x4+x5);
    x4 = x8 + (W1-W7)*x4;
    x5 = x8 - (W1+W7)*x5;
    x8 = W3*(x6+x7);
    x6 = x8 - (W3-W5)*x6;
    x7 = x8 - (W3+W5)*x7;

    /* second stage */
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2);
    x2 = x1 - (W2+W6)*x2;
    x3 = x1 + (W2-W6)*x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    /* third stage */
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (c181*(x4+x5)+c128)>>8;
    x4 = (c181*(x4-x5)+c128)>>8;

    /* fourth stage */
    blk[0] = ((x7+x1)>>8).cast<dfc::sint16>();
    blk[1] = ((x3+x2)>>8).cast<dfc::sint16>();
    blk[2] = ((x0+x4)>>8).cast<dfc::sint16>();
    blk[3] = ((x8+x6)>>8).cast<dfc::sint16>();
    blk[4] = ((x8-x6)>>8).cast<dfc::sint16>();
    blk[5] = ((x0-x4)>>8).cast<dfc::sint16>();
    blk[6] = ((x3-x2)>>8).cast<dfc::sint16>();
    blk[7] = ((x7-x1)>>8).cast<dfc::sint16>();
  }

  /* column (vertical) IDCT
   *
   *             7                         pi         1
   * dst[8*k] = sum c[l] * src[8*l] * cos( -- * ( k + - ) * l )
   *            l=0                        8          2
   *
   * where: c[0]    = 1/1024
   *        c[1..7] = (1/1024)*sqrt(2)
   */
  void idctcol(std::array<dfc::stream<dfc::sint16>, 64> blk, std::size_t i) {
    dfc::stream<dfc::sint32> x0, x1, x2, x3, x4, x5, x6, x7, x8;

    /* TODO: shortcut */
    x1 = blk[8*4+i].cast<dfc::sint32>()<<8;
    x2 = blk[8*6+i].cast<dfc::sint32>();
    x3 = blk[8*2+i].cast<dfc::sint32>();
    x4 = blk[8*1+i].cast<dfc::sint32>();
    x5 = blk[8*7+i].cast<dfc::sint32>();
    x6 = blk[8*5+i].cast<dfc::sint32>();
    x7 = blk[8*3+i].cast<dfc::sint32>();

    x0 = (blk[8*0+i].cast<dfc::sint32>()<<8) + c8192;

    /* first stage */
    x8 = W7*(x4+x5) + c4;
    x4 = (x8+(W1-W7)*x4)>>3;
    x5 = (x8-(W1+W7)*x5)>>3;
    x8 = W3*(x6+x7) + c4;
    x6 = (x8-(W3-W5)*x6)>>3;
    x7 = (x8-(W3+W5)*x7)>>3;
    
    /* second stage */
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2) + c4;
    x2 = (x1-(W2+W6)*x2)>>3;
    x3 = (x1+(W2-W6)*x3)>>3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    
    /* third stage */
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (c181*(x4+x5)+c128)>>8;
    x4 = (c181*(x4-x5)+c128)>>8;
    
    /* fourth stage */
    blk[8*0+i] = iclp((x7+x1)>>14);
    blk[8*1+i] = iclp((x3+x2)>>14);
    blk[8*2+i] = iclp((x0+x4)>>14);
    blk[8*3+i] = iclp((x8+x6)>>14);
    blk[8*4+i] = iclp((x8-x6)>>14);
    blk[8*5+i] = iclp((x0-x4)>>14);
    blk[8*6+i] = iclp((x3-x2)>>14);
    blk[8*7+i] = iclp((x7-x1)>>14);
  }

  dfc::typed<dfc::sint16>& iclp(dfc::typed<dfc::sint32> &in) {
    // FIXME:
    return in.cast<dfc::sint16>();
  }
};

dfc::value<dfc::sint32> IDCT::c4(4);
dfc::value<dfc::sint32> IDCT::c128(128);
dfc::value<dfc::sint32> IDCT::c181(181);
dfc::value<dfc::sint32> IDCT::c8192(8192);
dfc::value<dfc::sint32> IDCT::W1(2841);
dfc::value<dfc::sint32> IDCT::W2(2676);
dfc::value<dfc::sint32> IDCT::W3(2408);
dfc::value<dfc::sint32> IDCT::W5(1609);
dfc::value<dfc::sint32> IDCT::W6(1108);
dfc::value<dfc::sint32> IDCT::W7(565);

void dfcIdctTest() {
  dfc::params args;
  IDCT kernel(args);

  std::shared_ptr<eda::hls::model::Model> model =
    eda::hls::parser::dfc::Builder::get().create("IDCT");
  std::cout << *model << std::endl;

  std::ofstream output("dfc_idct_test.dot");
  eda::hls::model::printDot(output, *model);
  output.close();
}

TEST(DfcTest, DfcIdctTest) {
  dfcIdctTest();
}
