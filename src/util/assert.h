//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <iostream>

#define uassert(cond, message) \
  do {\
    if (!(cond)) {\
      std::cerr << message;\
      assert(cond);\
    }\
  } while (false)