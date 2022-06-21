//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "CLI/CLI.hpp"

#include <string>
#include <vector>

struct RtlOptions final {
  RtlOptions() = delete;

  RtlOptions(CLI::App &app) {
    options = app.add_subcommand("rtl", "Logical synthesis");

    // Input file(s).
    options->allow_extras();
  }

  std::vector<std::string> files() const {
    return options->remaining();
  }

private:
  CLI::App *options;
};

struct HlsOptions final {
  HlsOptions() = delete;

  HlsOptions(CLI::App &app) {
    options = app.add_subcommand("hls", "High-level synthesis");

    options->add_option("--output-dir",  outDir,  "Output directory")
           ->expected(1);
    options->add_option("--output-dot",  outDot,  "Output DOT file")
           ->expected(0, 1);
    options->add_option("--output-mlir", outMlir, "Output MLIR file")
           ->expected(1);
    options->add_option("--output-vlog", outVlog, "Output Verilog file")
           ->expected(1);
    options->add_option("--output-test", outTest, "Output test file")
           ->expected(0, 1);

    // Input file(s).
    options->allow_extras();
  }

  std::vector<std::string> files() const {
    return options->remaining();
  }

  std::string outDir;
  std::string outDot;
  std::string outMlir;
  std::string outVlog;
  std::string outTest;

private:
  CLI::App *options;
};

struct Options final {
  Options() = delete;

  Options(const std::string &title,
          const std::string &version,
          int argc, char **argv): app(title), rtl(app), hls(app) {
    app.set_help_all_flag("--help-all", "Print help");
    app.set_version_flag("--version", version, "Print version");

    app.parse(argc, argv);
  }

private:
  CLI::App app;

public:
  RtlOptions rtl;
  HlsOptions hls;
};
