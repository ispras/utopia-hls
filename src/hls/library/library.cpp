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

std::shared_ptr<MetaElement> Library::find(const NodeType &nodetype) {
  auto metaElement = find(nodetype.name);

  if (metaElement == nullptr) {
    // FIXME
    metaElement = MetaElementMock::create(nodetype);
    library.push_back(metaElement);
  }

  return metaElement;
}

std::shared_ptr<MetaElement> Library::find(const std::string &name) const {
  std::string lowerCaseName = name;
  unsigned x = 0;
  while(name[x]) {
    lowerCaseName[x] = tolower(lowerCaseName[x]);
    x++;
  }

  const auto i = std::find_if(library.begin(), library.end(),
    [&lowerCaseName](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == lowerCaseName;
    });

  return i == library.end() ? nullptr : *i;
}

} // namespace eda::hls::library
