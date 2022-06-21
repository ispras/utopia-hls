//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "CLI/CLI.hpp"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>

class AppOptions {
public:
  AppOptions() = delete;

  AppOptions(const std::string &title):
      isRoot(true), options(new CLI::App(title)) {}

  AppOptions(AppOptions &parent,
             const std::string &cmd,
             const std::string &desc):
      isRoot(false), options(parent.options->add_subcommand(cmd, desc)) {}

  virtual ~AppOptions() {
    if (isRoot) { delete options; }
  }

  std::string toConfig() const {
    return toConfig(options);
  }

protected:
  const bool isRoot;
  CLI::App *options;

private:
  std::string toConfig(const CLI::App *app) const {
    nlohmann::json json;

    for (const auto *opt : app->get_options({})) {
      if (!opt->get_lnames().empty() && opt->get_configurable()) {
        const auto name = opt->get_lnames()[0];

        if (opt->get_type_size() != 0) {
          if (opt->count() == 1) {
            json[name] = opt->results().at(0);
          } else if (opt->count() > 1) {
            json[name] = opt->results();
          } else if (!opt->get_default_str().empty()) {
            json[name] = opt->get_default_str();
          }
        } else if (opt->count() == 1) {
          json[name] = true;
        } else if (opt->count() > 1) {
          json[name] = opt->count();
        } else if (opt->count() == 0) {
          json[name] = false;
        }
      }
    }

    for(const auto *cmd : app->get_subcommands({})) {
      json[cmd->get_name()] = nlohmann::json(toConfig(cmd));
    }

    return json.dump(2);
  }
};

struct RtlOptions final : public AppOptions {
  RtlOptions(AppOptions &parent):
      AppOptions(parent, "rtl", "Logical synthesis") {

    // Input file(s).
    options->allow_extras();
  }

  std::vector<std::string> files() const {
    return options->remaining();
  }
};

struct HlsOptions final : public AppOptions {
  HlsOptions(AppOptions &parent):
      AppOptions(parent, "hls", "High-level synthesis") {

    // Named options.
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
};

struct Options final : public AppOptions {
  Options(const std::string &title,
          const std::string &version,
          int argc, char **argv):
      AppOptions(title), rtl(*this), hls(*this) {

    // Top-level options.
    options->set_help_all_flag("--help-all", "Print help");
    options->set_version_flag("--version", version, "Print version");

    options->parse(argc, argv);
  }

  RtlOptions rtl;
  HlsOptions hls;
};
