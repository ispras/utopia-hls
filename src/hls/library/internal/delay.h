#include "hls/library/ipxact_parser.h"
#include "hls/library/library.h"
#include "hls/library/library_mock.h"
#include "util/assert.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

namespace eda::hls::library {

struct Internal : public MetaElement {
  Internal(const std::string &name,
           const Parameters &params,
           const std::vector<Port> &ports)
  MetaElementMock(name(name),
                  params(params),
                  ports(ports)) {}
  std::unique_ptr<Element> construct(
        const Parameters &params) const override;
  virtual void estimate(
      const Parameters &params, Indicators &indicators) const override;
};

} // namespace eda::hls::library
