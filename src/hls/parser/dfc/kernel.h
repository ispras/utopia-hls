//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "stream.h"
#include "utils/assert.h"
#include "utils/string.h"

#include <string>

#define DFC_KERNEL(UserKernel) \
  struct UserKernel: public dfc::user_kernel<UserKernel>

#define DFC_KERNEL_DTOR(UserKernel) \
  public: virtual ~UserKernel()

#define DFC_KERNEL_ACTIVATE \
  eda::hls::parser::dfc::Builder::get().activateKernel()

#define DFC_KERNEL_DEACTIVATE \
  eda::hls::parser::dfc::Builder::get().deactivateKernel()

#define DFC_CREATE_KERNEL_FUNCTION(UserKernel) \
  public: \
  static std::shared_ptr<UserKernel> create(const std::string &name) { \
      return std::shared_ptr<UserKernel>(new UserKernel(name)); }

#define DFC_CREATE_KERNEL(UserKernel) \
  UserKernel::create(#UserKernel);

#define DFC_KERNEL_CTOR(UserKernel) \
  private: \
  UserKernel(const std::string &name): \
    dfc::user_kernel<UserKernel>(name)

namespace dfc {

class kernel {
public:
  const std::string name;

protected:
  kernel(const std::string &name);
  kernel(): kernel(eda::utils::unique_name("kernel")) {}
  virtual ~kernel() {}
};

template<typename UserKernel>
class user_kernel: public kernel {
protected:
  user_kernel(const std::string &id): kernel(id) {}
  virtual ~user_kernel() {}
};

} // namespace dfc