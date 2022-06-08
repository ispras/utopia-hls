#include "hls/library/element_internal.h"
#include "hls/library/library.h"

#include <memory>

namespace eda::hls::library {

struct Delay final : public ElementInternal {
  static constexpr const char *name  = "delay";
  static constexpr const char *width = "width";
  static constexpr const char *depth = "stages";

  Delay(): ElementInternal(name, {}, {}) {}
};

} // namespace eda::hls::library
