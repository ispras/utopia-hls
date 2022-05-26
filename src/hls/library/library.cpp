//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "hls/library/library_mock.h"
#include "util/assert.h"

#include <algorithm>

namespace eda::hls::library {

void Library::initialize() {
  // TODO:
}

std::shared_ptr<MetaElement> Library::find(const NodeType &nodetype) {
  return find(nodetype.name);
}

std::shared_ptr<MetaElement> Library::find(const std::string &name) {
  const auto i = std::find_if(library.begin(), library.end(),
    [&name](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == name;
    });

  if (i == library.end()) {
    // FIXME
    auto metaElement = MetaElementMock::create(name);
    library.push_back(metaElement);
    return metaElement;
  }

  return *i;
}

} // namespace eda::hls::library
