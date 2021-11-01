//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace eda::util {

template <typename T>
class Singleton {
public:
  static T& get() {
    static std::unique_ptr<T> instance(new T());
    return *instance;
  }

  Singleton(const Singleton &other) = delete;
  Singleton& operator=(const Singleton &other) = delete;

protected:
  Singleton() {}

private:
  inline static T *instance = nullptr;
};

} // namespace eda::util
