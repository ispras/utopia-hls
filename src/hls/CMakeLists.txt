add_subdirectory(parser/hil)
add_subdirectory(model/mlir)

add_library(HLS OBJECT
  compiler/compiler.cpp
  debugger/debugger.cpp
  library/element_generator.cpp
  library/element_core.cpp
  library/element_internal.cpp
  library/internal/cast.cpp
  library/internal/clip.cpp
  library/internal/const.cpp
  library/internal/default.cpp
  library/internal/delay.cpp
  library/internal/dup.cpp
  library/internal/merge.cpp
  library/internal/mux.cpp
  library/internal/split.cpp
  library/ipxact_parser.cpp
  library/library.cpp
  mapper/config/hwconfig.cpp
  mapper/mapper.cpp
  model/model.cpp
  model/transform.cpp
  model/printer.cpp
  parser/dfc/kernel.cpp
  parser/dfc/stream.cpp
  parser/dfc/internal/builder.cpp
  scheduler/dijkstra.h
  scheduler/flow_solver.cpp
  scheduler/latency_balancer_base.h
  scheduler/latency_solver.cpp
  scheduler/lp_helper.cpp
  scheduler/optimizers/abstract_optimizer.cpp
  scheduler/optimizers/simple_optimizer.cpp
  scheduler/optimizers/simulated_annealing_optimizer.cpp
  scheduler/param_optimizer.h
  scheduler/topological_balancer.cpp
)

target_include_directories(HLS PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(HLS
  PUBLIC
    MLIRHIL
    XercesC::XercesC
    LpSolve::LpSolve
    Z3::Z3
    CTemplate::nothreads
    ${STDCXXFS_LIBRARY}
)

add_library(Utopia::HLS ALIAS HLS)
