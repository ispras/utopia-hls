//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "CLI/CLI.hpp"
#include "gate/premapper/premapper.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Json = nlohmann::json;

NLOHMANN_JSON_SERIALIZE_ENUM( eda::gate::premapper::PreBasis, {
    {eda::gate::premapper::AIG, "aig"},
    {eda::gate::premapper::MIG, "mig"},
    {eda::gate::premapper::XAG, "xag"},
    {eda::gate::premapper::XMG, "xmg"},
})

class AppOptions {
public:
  AppOptions() = delete;

  AppOptions(const std::string &title,
             const std::string &version):
      isRoot(true), options(new CLI::App(title)) {}

  AppOptions(AppOptions &parent,
             const std::string &cmd,
             const std::string &desc):
      isRoot(false), options(parent.options->add_subcommand(cmd, desc)) {}

  virtual ~AppOptions() {
    if (isRoot) { delete options; }
  }

  void parse(int argc, char **argv) {
    options->parse(argc, argv);
  }

  virtual void fromJson(Json json) {
    // TODO: Default implementation.
  }

  virtual Json toJson() const {
    return toJson(options);
  }

  virtual void read(std::istream &in) {
    auto json = Json::parse(in);
    fromJson(json);
  }

  virtual void save(std::ostream &out) const {
    auto json = toJson();
    out << json;
  }

  void read(const std::string &config) {
    std::ifstream in(config);
    if (in.good()) {
      read(in);
    }
  }

  void save(const std::string &config) const {
    std::ofstream out(config);
    if (out.good()) {
      save(out);
    }
  }

protected:
  static std::string cli(const std::string &option) {
    return "--" + option;
  }

  static void get(Json json, const std::string &key, std::string &value) {
    if (json.contains(key)) {
      value = json[key].get<std::string>();
    }
  }

  template<class T>
  static void get(Json json, const std::string &key, T &value) {
    if (json.contains(key)) {
      value = json[key].get<T>();
    }
  }

  Json toJson(const CLI::App *app) const {
    Json json;

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
      json[cmd->get_name()] = Json(toJson(cmd));
    }

    return json;
  }

  const bool isRoot;
  CLI::App *options;
};

struct RtlOptions final : public AppOptions {

  using PreBasis = eda::gate::premapper::PreBasis;

  static constexpr const char *ID = "rtl";

  static constexpr const char *PREMAP_BASIS  = "premap-basis";
  static constexpr const char *PREMAP_LIB    = "premap-lib";

  const std::map<std::string, PreBasis> preBasisMap {
    {"aig", PreBasis::AIG},
    {"mig", PreBasis::MIG},
    {"xag", PreBasis::XAG},
    {"xmg", PreBasis::XMG}
  };

  RtlOptions(AppOptions &parent):
      AppOptions(parent, ID, "Logical synthesis") {

    // Named options.
    options->add_option(cli(PREMAP_BASIS), preBasis, "Premapper basis")
           ->expected(1)
           ->transform(CLI::CheckedTransformer(preBasisMap, CLI::ignore_case));
    options->add_option(cli(PREMAP_LIB), preLib, "Premapper library")
           ->expected(1);

    // Input file(s).
    options->allow_extras();
  }

  std::vector<std::string> files() const {
    return options->remaining();
  }

  void fromJson(Json json) override {
    get(json, PREMAP_BASIS, preBasis);
    get(json, PREMAP_LIB, preLib);
  }

  eda::gate::premapper::PreBasis preBasis = PreBasis::AIG;
  std::string preLib;
};

struct HlsOptions final : public AppOptions {
  static constexpr const char *ID = "hls";

  static constexpr const char *OUTPUT_DIR  = "output-dir";
  static constexpr const char *OUTPUT_DOT  = "output-dot";
  static constexpr const char *OUTPUT_MLIR = "output-mlir";
  static constexpr const char *OUTPUT_LIB  = "output-lib";
  static constexpr const char *OUTPUT_TOP  = "output-top";
  static constexpr const char *OUTPUT_TEST = "output-test";

  HlsOptions(AppOptions &parent):
      AppOptions(parent, ID, "High-level synthesis") {

    // Named options.
    options->add_option(cli(OUTPUT_DIR),  outDir,  "Output directory")
           ->expected(1);
    options->add_option(cli(OUTPUT_DOT),  outDot,  "Output DOT file")
           ->expected(0, 1);
    options->add_option(cli(OUTPUT_MLIR), outMlir, "Output MLIR file")
           ->expected(1);
    options->add_option(cli(OUTPUT_LIB),  outLib,  "Output Verilog library file")
           ->expected(1);
    options->add_option(cli(OUTPUT_TOP),  outTop,  "Output Verilog top file")
          ->expected(1);
    options->add_option(cli(OUTPUT_TEST), outTest, "Output test file")
           ->expected(0, 1);

    // Input file(s).
    options->allow_extras();
  }

  std::vector<std::string> files() const {
    return options->remaining();
  }

  void fromJson(Json json) override {
    get(json, OUTPUT_DIR,  outDir);
    get(json, OUTPUT_DOT,  outDot);
    get(json, OUTPUT_MLIR, outMlir);
    get(json, OUTPUT_LIB,  outLib);
    get(json, OUTPUT_TOP,  outTop);
    get(json, OUTPUT_TEST, outTest);
  }

  std::string outDir;
  std::string outDot;
  std::string outMlir;
  std::string outLib;
  std::string outTop;
  std::string outTest;
};

struct Options final : public AppOptions {
  Options(const std::string &title,
          const std::string &version):
      AppOptions(title, version), rtl(*this), hls(*this) {

    // Top-level options.
    options->set_help_all_flag("-H,--help-all", "Print the extended help message and exit");
    options->set_version_flag("-v,--version", version, "Print the tool version");
  }

  void initialize(const std::string &config, int argc, char **argv) {
    // Read the JSON configuration.
    read(config);
    // Command line is of higher priority.
    parse(argc, argv);
  }

  int exit(const CLI::Error &e) const {
    return options->exit(e);
  }

  void fromJson(Json json) override {
    rtl.fromJson(json[RtlOptions::ID]);
    hls.fromJson(json[HlsOptions::ID]);
  }

  RtlOptions rtl;
  HlsOptions hls;
};
