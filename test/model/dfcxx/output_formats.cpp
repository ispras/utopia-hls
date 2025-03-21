//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "polynomial2/polynomial2.h"

#include "gtest/gtest.h"

static const Polynomial2 kernel;

static const dfcxx::DFLatencyConfig config =
    dfcxx::DFLatencyConfig({{dfcxx::ADD_INT, 2}, {dfcxx::MUL_INT, 3}}, {});

TEST(DFCXXOutputFormats, SystemVerilog) {
  Polynomial2 kernel;
  dfcxx::DFOutputPaths paths = {
    {dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}

TEST(DFCXXOutputFormats, SystemVerilogLibrary) {
  Polynomial2 kernel;
  dfcxx::DFOutputPaths paths = {
    {dfcxx::OutputFormatID::SVLibrary, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}

TEST(DFCXXOutputFormats, UnscheduledDFCIR) {
  Polynomial2 kernel;
  dfcxx::DFOutputPaths paths = {
    {dfcxx::OutputFormatID::UnscheduledDFCIR, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}

TEST(DFCXXOutputFormats, FIRRTL) {
  Polynomial2 kernel;
  dfcxx::DFOutputPaths paths = {
    {dfcxx::OutputFormatID::FIRRTL, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}

TEST(DFCXXOutputFormats, DOT) {
  Polynomial2 kernel;
  dfcxx::DFOutputPaths paths = {
    {dfcxx::OutputFormatID::DOT, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}

TEST(DFCXXOutputFormats, All) {
  Polynomial2 kernel;
  dfcxx::DFOutputPaths paths = {
    {dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE},
    {dfcxx::OutputFormatID::SVLibrary, NULLDEVICE},
    {dfcxx::OutputFormatID::UnscheduledDFCIR, NULLDEVICE},
    {dfcxx::OutputFormatID::ScheduledDFCIR, NULLDEVICE},
    {dfcxx::OutputFormatID::FIRRTL, NULLDEVICE},
    {dfcxx::OutputFormatID::DOT, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}
