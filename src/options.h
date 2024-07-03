//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

// Is needed for DFCxx output formats definitions and avaiable op. types.
#include "dfcxx/typedefs.h"

#include "CLI/CLI.hpp"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// CLI macro definitions

#define CLI_FLAG(FLAG) "-" FLAG
#define CLI_FLAG_E(FLAG) CLI_FLAG(FLAG)

#define CLI_ARG(ARG) "--" ARG
#define CLI_ARG_E(ARG) CLI_ARG(ARG)

//===----------------------------------------------------------------------===//
// JSON config fields' definitions

#define HLS_ID_JSON "hls"
#define CONFIG_JSON "config"
#define ASAP_SCHEDULER_JSON "asap_scheduler"
#define LP_SCHEDULER_JSON "lp_scheduler"
#define SV_OUT_JSON "sv_out"

//===----------------------------------------------------------------------===//
// CLI args/flags definitions

#define HLS_CMD "hls"
#define CONFIG_ARG CLI_ARG("config")
#define SCHEDULER_GROUP "scheduler"
#define ASAP_SCHEDULER_FLAG CLI_FLAG("a")
#define LP_SCHEDULER_FLAG CLI_FLAG("l")
#define OUTPUT_GROUP "output"
#define SV_OUT_ARG CLI_ARG("sv_out")

//===----------------------------------------------------------------------===//

using Json = nlohmann::json;

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

struct HlsOptions final : public AppOptions {

  HlsOptions(AppOptions &parent):
      AppOptions(parent, HLS_CMD, "High-level synthesis"),
      // Initialize the output paths vector for every possible format.
      outNames(OUTPUT_FORMATS_COUNT) {
      
    // Named options.
    options->add_option(CONFIG_ARG, latConfigFile, "JSON latency configuration path")
        ->expected(1);
    
    auto schedGroup = options->add_option_group(SCHEDULER_GROUP);
    schedGroup->add_flag(ASAP_SCHEDULER_FLAG, asapScheduler, "Use greedy as-soon-as-possible scheduler");
    schedGroup->add_flag(LP_SCHEDULER_FLAG,   lpScheduler,   "Use Linear Programming scheduler");
    schedGroup->require_option(1); 
    auto outputGroup = options->add_option_group(OUTPUT_GROUP);
    outputGroup->add_option(SV_OUT_ARG, outNames[SV_OUT_ID], "File path for SystemVerilog output");
    outputGroup->require_option();
  }

  void fromJson(Json json) override {
    get(json, CONFIG_JSON,         latConfigFile);
    get(json, ASAP_SCHEDULER_JSON, asapScheduler);
    get(json, LP_SCHEDULER_JSON,   lpScheduler);
    get(json, SV_OUT_JSON,         outNames[SV_OUT_ID]);
  }

  std::string latConfigFile;
  DFLatencyConfig latConfig;
  std::vector<std::string> outNames;
  bool asapScheduler;
  bool lpScheduler;
};

struct Options final : public AppOptions {
  Options(const std::string &title,
          const std::string &version):
      AppOptions(title, version), hls(*this) {

    // Top-level options.
    options->set_help_all_flag("-H,--help-all", "Print the extended help message and exit");
    options->set_version_flag("-v,--version", version, "Print the tool version");
  }
  
  dfcxx::Ops convertFieldToEnum(const std::string field) {
    if (field == "ADD_INT")         { return dfcxx::ADD_INT; }          else
    if (field == "ADD_FLOAT")       { return dfcxx::ADD_FLOAT; }        else
    if (field == "SUB_INT")         { return dfcxx::SUB_INT; }          else
    if (field == "SUB_FLOAT")       { return dfcxx::SUB_FLOAT; }        else
    if (field == "MUL_INT")         { return dfcxx::MUL_INT; }          else
    if (field == "MUL_FLOAT")       { return dfcxx::MUL_FLOAT; }        else
    if (field == "DIV_INT")         { return dfcxx::DIV_INT; }          else
    if (field == "DIV_FLOAT")       { return dfcxx::DIV_FLOAT; }        else
    if (field == "AND_INT")         { return dfcxx::AND_INT; }          else
    if (field == "AND_FLOAT")       { return dfcxx::AND_FLOAT; }        else
    if (field == "OR_INT")          { return dfcxx::OR_INT; }           else
    if (field == "OR_FLOAT")        { return dfcxx::OR_FLOAT; }         else
    if (field == "XOR_INT")         { return dfcxx::XOR_INT; }          else
    if (field == "XOR_FLOAT")       { return dfcxx::XOR_FLOAT; }        else
    if (field == "NOT_INT")         { return dfcxx::NOT_INT; }          else
    if (field == "NOT_FLOAT")       { return dfcxx::NOT_FLOAT; }        else
    if (field == "NEG_INT")         { return dfcxx::NEG_INT; }          else
    if (field == "NEG_FLOAT")       { return dfcxx::NEG_FLOAT; }        else
    if (field == "LESS_INT")        { return dfcxx::LESS_INT; }         else
    if (field == "LESS_FLOAT")      { return dfcxx::LESS_FLOAT; }       else
    if (field == "LESSEQ_INT")      { return dfcxx::LESSEQ_INT; }       else
    if (field == "LESSEQ_FLOAT")    { return dfcxx::LESSEQ_FLOAT; }     else
    if (field == "GREATER_INT")     { return dfcxx::GREATER_INT; }      else
    if (field == "GREATER_FLOAT")   { return dfcxx::GREATER_FLOAT; }    else
    if (field == "GREATEREQ_INT")   { return dfcxx::GREATEREQ_INT; }    else
    if (field == "GREATEREQ_FLOAT") { return dfcxx::GREATEREQ_FLOAT; }  else
    if (field == "EQ_INT")          { return dfcxx::EQ_INT; }           else
    if (field == "EQ_FLOAT")        { return dfcxx::EQ_FLOAT; }         else
    if (field == "NEQ_INT")         { return dfcxx::NEQ_INT; }          else
    if (field == "NEQ_FLOAT")       { return dfcxx::NEQ_FLOAT; }        else
    return dfcxx::ADD_INT;
  }

  void parseLatencyConfig(const std::string config) {
    std::ifstream in(config);
    if (!in.good()) { return; }
    auto json = Json::parse(in);
    for (auto &[key, val] : json.items()) {
      hls.latConfig[convertFieldToEnum(key)] = val;
    }
  }

  void initialize(const std::string &config, int argc, char **argv) {
    // Read the JSON configuration.
    read(config);
    // Command line is of higher priority.
    parse(argc, argv);
    // Parse latency configuration.
    parseLatencyConfig(hls.latConfigFile);
  }

  int exit(const CLI::Error &e) const {
    return options->exit(e);
  }

  void fromJson(Json json) override {
    hls.fromJson(json[HLS_ID_JSON]);
  }

  HlsOptions hls;
};
