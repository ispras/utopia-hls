## SPDX-License-Identifier: Apache-2.0

set(PASSES 
  conversions/DFCIRToFIRRTLPass.cpp
  conversions/DFCIRASAPSchedulerPass.cpp
  conversions/DFCIRLinearSchedulerPass.cpp
  conversions/DFCIRLPUtils.cpp
  conversions/FIRRTLStubsGeneratorPass.cpp
)

set(CONV_LIBS
  ${PASSES}
  conversions/DFCIRPassesUtils.cpp
  PARENT_SCOPE
)
