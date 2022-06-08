#include "hls/library/element_internal.h"
#include "hls/library/library.h"

#include <memory>

namespace eda::hls::library {

struct Delay final : public ElementInternal {
  static constexpr std::string name  = "delay";
  static constexpr std::string width = "WIDTH";
  static constexpr std::string depth = "DEPTH";

  static Parameters getParams() {
    Parameters params;
    params.add(Parameter(width));
    params.add(Parameter(depth));
    return params;
  }

  static std::vector<Port>& getPorts() {
    std::vector<Port> ports;
    return ports;
  }

  Delay(): ElementInternal(name, getParams(), getPorts()) {}
};

} // namespace eda::hls::library
