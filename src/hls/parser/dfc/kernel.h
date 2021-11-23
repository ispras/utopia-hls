//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "util/string.h"

#include <string>

#define DFC_KERNEL(UserKernel) \
  struct UserKernel: public dfc::user_kernel<UserKernel>

#define DFC_KERNEL_CTOR(UserKernel) \
  UserKernel(const dfc::params &args): dfc::user_kernel<UserKernel>(#UserKernel, args)

#define DFC_KERNEL_DTOR(UserKernel) \
  virtual ~UserKernel()

namespace dfc {

struct params final {
  // TODO:
};

class kernel {
public:
  const std::string name;

protected:
  kernel(const std::string &name, const params &args);
  kernel(const params &args): kernel(eda::utils::unique_name("kernel"), args) {}
  virtual ~kernel() {}
};

template<typename UserKernel>
class user_kernel: public kernel {
protected:
  user_kernel(const std::string &id, const params &args): kernel(id, args) {}
  virtual ~user_kernel() {}
};

} // namespace dfc
