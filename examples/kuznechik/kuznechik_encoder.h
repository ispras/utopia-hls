//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

#include <cassert>
#include <vector>

#define C(NUM) constant.var(type, uint64_t(NUM))

using dfcxx::DFType;
using dfcxx::DFVariable;

class KuznechikEncoder : public dfcxx::Kernel {
public:
  std::string_view getName() const override {
    return "KuznechikEncoder";
  }

  ~KuznechikEncoder() override = default;

  using DFType = dfcxx::DFType;
  using DFVariable = dfcxx::DFVariable;

  DFVariable kuznechikTablePermut(DFVariable val) {
    const DFType type = dfUInt(8);
    return control.mux(val, {
      C(252), C(238), C(221), C(17), C(207), C(110), C(49), C(22),
      C(251), C(196), C(250), C(218), C(35), C(197), C(4), C(77),
      C(233), C(119), C(240), C(219), C(147), C(46), C(153), C(186),
      C(23), C(54), C(241), C(187), C(20), C(205), C(95), C(193),
      C(249), C(24), C(101), C(90), C(226), C(92), C(239), C(33),
      C(129), C(28), C(60), C(66), C(139), C(1), C(142), C(79),
      C(5), C(132), C(2), C(174), C(227), C(106), C(143), C(160),
      C(6), C(11), C(237), C(152), C(127), C(212), C(211), C(31),
      C(235), C(52), C(44), C(81), C(234), C(200), C(72), C(171),
      C(242), C(42), C(104), C(162), C(253), C(58), C(206), C(204),
      C(181), C(112), C(14), C(86), C(8), C(12), C(118), C(18),
      C(191), C(114), C(19), C(71), C(156), C(183), C(93), C(135),
      C(21), C(161), C(150), C(41), C(16), C(123), C(154), C(199),
      C(243), C(145), C(120), C(111), C(157), C(158), C(178), C(177),
      C(50), C(117), C(25), C(61), C(255), C(53), C(138), C(126),
      C(109), C(84), C(198), C(128), C(195), C(189), C(13), C(87),
      C(223), C(245), C(36), C(169), C(62), C(168), C(67), C(201),
      C(215), C(121), C(214), C(246), C(124), C(34), C(185), C(3),
      C(224), C(15), C(236), C(222), C(122), C(148), C(176), C(188),
      C(220), C(232), C(40), C(80), C(78), C(51), C(10), C(74),
      C(167), C(151), C(96), C(115), C(30), C(0), C(98), C(68),
      C(26), C(184), C(56), C(130), C(100), C(159), C(38), C(65),
      C(173), C(69), C(70), C(146), C(39), C(94), C(85), C(47),
      C(140), C(163), C(165), C(125), C(105), C(213), C(149), C(59),
      C(7), C(88), C(179), C(64), C(134), C(172), C(29), C(247),
      C(48), C(55), C(107), C(228), C(136), C(217), C(231), C(137),
      C(225), C(27), C(131), C(73), C(76), C(63), C(248), C(254),
      C(141), C(83), C(170), C(144), C(202), C(216), C(133), C(97),
      C(32), C(113), C(103), C(164), C(45), C(43), C(9), C(91),
      C(203), C(155), C(37), C(208), C(190), C(229), C(108), C(82),
      C(89), C(166), C(116), C(210), C(230), C(244), C(180), C(192),
      C(209), C(102), C(175), C(194), C(57), C(75), C(99), C(182)
    });
  }

  DFVariable kuznechikSPermut(DFVariable value) {
    assert(value.getTotalBits() == 128);
    
    DFVariable substituted = kuznechikTablePermut(value(127, 120));
    for (int i = 1; i < 16; ++i) {
      int currSInd = 127 - i*8;
      substituted =
          substituted.cat(kuznechikTablePermut(value(currSInd, currSInd - 7)));
    }

    assert(substituted.getTotalBits() == 128);

    return substituted;
  }

  DFVariable kuznechikMulGf(DFVariable left, DFVariable right) {
    const DFType type = left.getType();
    assert(type.getTotalBits() == 8);
    DFVariable c0 = constant.var(type, uint64_t(0));
    DFVariable c195 = constant.var(type, uint64_t(195));
    DFVariable currValue = c0;

    for (int i = 0; i < 7; ++i) {
      DFVariable isBitSet = right(0, 0);
      currValue = currValue ^ control.mux(isBitSet, {c0, left});
      DFVariable aboutToOverflow = left(7, 7);
      DFVariable muxed = control.mux(aboutToOverflow, {c0, c195});
      left = (left << 1) ^ muxed;
      right = right >> 1;
    }

    DFVariable isBitSet = right(0, 0);
    currValue = currValue ^ control.mux(isBitSet, {c0, left});

    return currValue;
  }

  DFVariable kuznechikLinearMapping(DFVariable val) {
    const DFType type = dfUInt(8);
    DFVariable result =
        kuznechikMulGf(constant.var(type, uint64_t(148)),val(127, 120));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(32)), val(119, 112));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(133)), val(111, 104));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(16)), val(103, 96));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(194)), val(95, 88));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(192)), val(87, 80));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(1)), val(79, 72));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(251)), val(71, 64));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(1)), val(63, 56));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(192)), val(55, 48));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(194)), val(47, 40));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(16)), val(39, 32));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(133)), val(31, 24));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(32)), val(23, 16));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(148)), val(15, 8));
    result ^= kuznechikMulGf(constant.var(type, uint64_t(1)), val(7, 0));
    return result;
  }

  DFVariable kuznechikRPermut(DFVariable val) {
    return kuznechikLinearMapping(val).cat(val(127, 8));
  }

  DFVariable kuznechikLinearPermut(DFVariable val) {
    DFVariable currVal = val;
    for (int i = 0; i < 16; ++i) {
      currVal = kuznechikRPermut(currVal);
    }
    return currVal;
  }

  std::vector<DFVariable> kuznechikGenConsts() {
    const DFType type = dfUInt(64);
    return {
      C(0x6ea276726c487ab8).cat(C(0x5d27bd10dd849401)),
      C(0xdc87ece4d890f4b3).cat(C(0xba4eb92079cbeb02)),
      C(0xb2259a96b4d88e0b).cat(C(0xe7690430a44f7f03)),
      C(0x7bcd1b0b73e32ba5).cat(C(0xb79cb140f2551504)),
      C(0x156f6d791fab511d).cat(C(0xeabb0c502fd18105)),
      C(0xa74af7efab73df16).cat(C(0xdd208608b9efe06)),
      C(0xc9e8819dc73ba5ae).cat(C(0x50f5b570561a6a07)),
      C(0xf6593616e6055689).cat(C(0xadfba18027aa2a08)),
      C(0x98fb40648a4d2c31).cat(C(0xf0dc1c90fa2ebe09)),
      C(0x2adedaf23e95a23a).cat(C(0x17b518a05e61c10a)),
      C(0x447cac8052ddd882).cat(C(0x4a92a5b083e5550b)),
      C(0x8d942d1d95e67d2c).cat(C(0x1a6710c0d5ff3f0c)),
      C(0xe3365b6ff9ae0794).cat(C(0x4740add0087bab0d)),
      C(0x5113c1f94d76899f).cat(C(0xa029a9e0ac34d40e)),
      C(0x3fb1b78b213ef327).cat(C(0xfd0e14f071b0400f)),
      C(0x2fb26c2c0f0aacd1).cat(C(0x993581c34e975410)),
      C(0x41101a5e6342d669).cat(C(0xc4123cd39313c011)),
      C(0xf33580c8d79a5862).cat(C(0x237b38e3375cbf12)),
      C(0x9d97f6babbd222da).cat(C(0x7e5c85f3ead82b13)),
      C(0x547f77277ce98774).cat(C(0x2ea93083bcc24114)),
      C(0x3add015510a1fdcc).cat(C(0x738e8d936146d515)),
      C(0x88f89bc3a47973c7).cat(C(0x94e789a3c509aa16)),
      C(0xe65aedb1c831097f).cat(C(0xc9c034b3188d3e17)),
      C(0xd9eb5a3ae90ffa58).cat(C(0x34ce2043693d7e18)),
      C(0xb7492c48854780e0).cat(C(0x69e99d53b4b9ea19)),
      C(0x56cb6de319f0eeb).cat(C(0x8e80996310f6951a)),
      C(0x6bcec0ac5dd77453).cat(C(0xd3a72473cd72011b)),
      C(0xa22641319aecd1fd).cat(C(0x835291039b686b1c)),
      C(0xcc843743f6a4ab45).cat(C(0xde752c1346ecff1d)),
      C(0x7ea1add5427c254e).cat(C(0x391c2823e2a3801e)),
      C(0x1003dba72e345ff6).cat(C(0x643b95333f27141f)),
      C(0x5ea7d8581e149b61).cat(C(0xf16ac1459ceda820))
    };
  }

  std::pair<DFVariable, DFVariable> kuznechikFMapping(DFVariable constVal,
                                                      DFVariable a1,
                                                      DFVariable a0) {
    return std::make_pair(kuznechikXSLPermut(constVal, a1) ^ a0, a1);
  }

  std::vector<DFVariable> kuznechikGenKeys(DFVariable k0, DFVariable k1) {
    std::vector<DFVariable> consts = kuznechikGenConsts();
    std::vector<DFVariable> keys;
    keys.push_back(k0);
    keys.push_back(k1);

    for (int i = 0; i < 4; ++i) {
      DFVariable currLeft =  keys[2 * i];
      DFVariable currRight = keys[2 * i + 1];
      for (int arrOffset = 0; arrOffset < 8; ++arrOffset) {
        auto leftAndRight =
            kuznechikFMapping(consts[8 * i + arrOffset], currLeft, currRight);
        currLeft = leftAndRight.first;
        currRight = leftAndRight.second;
      }
      keys.push_back(currLeft);
      keys.push_back(currRight);
    }
    return keys;
  }

  DFVariable kuznechikXSLPermut(DFVariable key, DFVariable value) {
    DFVariable addedKey = key ^ value;
    DFVariable permut = kuznechikSPermut(addedKey);
    return kuznechikLinearPermut(permut);
  }

  KuznechikEncoder() : dfcxx::Kernel() {
    const DFType ioType = dfUInt(128);

    DFVariable block = io.input("block", ioType);
    DFVariable key = io.input("key", dfUInt(256));

    std::vector<DFVariable> gKeys =
        kuznechikGenKeys(key(255, 128), key(127, 0));

    DFVariable currValue = block;
    for (int i = 0; i < 9; ++i) {
      currValue = kuznechikXSLPermut(gKeys[i], currValue);
    }

    currValue = currValue ^ gKeys[9];

    DFVariable encoded = io.output("encoded", ioType);
    encoded.connect(currValue);
  }
};
