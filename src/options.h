//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2025 ISP RAS (http://www.ispras.ru)
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
#define OUT_SV_JSON "out_sv"
#define OUT_SV_LIB_JSON "out_sv_lib"
#define OUT_UNSCHEDULED_DFCIR_JSON "out_dfcir"
#define OUT_SCHEDULED_DFCIR_JSON "out_scheduled_dfcir"
#define OUT_FIRRTL_JSON "out_firrtl"
#define OUT_DOT_JSON "out_dot"

#define DFCIR_OPS_ID_JSON "dfcir"
#define EXTERNAL_OPS_ID_JSON "external"

#define SIM_ID_JSON "sim"
#define SIM_IN_JSON "in"
#define SIM_OUT_JSON "out"

//===----------------------------------------------------------------------===//
// CLI args/flags definitions

#define HLS_CMD "hls"
#define SIM_CMD "sim"

#define CONFIG_ARG CLI_ARG("config")
#define CONFIG_ARG_DEFAULT "latency.json"

#define SCHEDULER_GROUP "scheduler"
#define ASAP_SCHEDULER_FLAG CLI_FLAG("a")
#define LP_SCHEDULER_FLAG CLI_FLAG("l")
#define PIPELINE_SCHEDULER_ARG CLI_ARG("pipeline")
#define OUTPUT_GROUP "output"

#define OUT_SV_ARG CLI_ARG("out-sv")
#define OUT_SV_ARG_DEFAULT "output.sv"

#define OUT_SV_LIB_ARG CLI_ARG("out-sv-lib")
#define OUT_SV_LIB_ARG_DEFAULT "output-lib.sv"

#define OUT_UNSCHEDULED_DFCIR_ARG CLI_ARG("out-dfcir")
#define OUT_UNSCHEDULED_DFCIR_ARG_DEFAULT "dfcir.mlir"

#define OUT_SCHEDULED_DFCIR_ARG CLI_ARG("out-scheduled-dfcir")
#define OUT_SCHEDULED_DFCIR_ARG_DEFAULT "scheduled-dfcir.mlir"

#define OUT_FIRRTL_ARG CLI_ARG("out-firrtl")
#define OUT_FIRRTL_ARG_DEFAULT "firrtl.mlir"

#define OUT_DOT_ARG CLI_ARG("out-dot")
#define OUT_DOT_ARG_DEFAULT "output.dot"

#define SIM_IN_ARG CLI_ARG("in")
#define SIM_IN_ARG_DEFAULT "sim.txt"

#define SIM_OUT_ARG CLI_ARG("out")
#define SIM_OUT_ARG_DEFAULT "sim_out.vcd"

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
  
  operator bool() const {
    return options->parsed();
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
      outNames(OUT_FORMAT_ID_INT(COUNT)) {

    // Named options.
    options
        ->add_option(CONFIG_ARG,
                     latencyCfgFile,
                     "JSON latency configuration path")
        ->default_str(CONFIG_ARG_DEFAULT)
        ->expected(0, 1);
    
    auto schedGroup = options->add_option_group(SCHEDULER_GROUP);
    schedGroup->add_flag(ASAP_SCHEDULER_FLAG,
                         "Use greedy as-soon-as-possible scheduler");
    schedGroup->add_flag(LP_SCHEDULER_FLAG,
                         "Use Linear Programming scheduler");
    schedGroup
        ->add_option(PIPELINE_SCHEDULER_ARG,
                     optionsCfg.stages,
                     "Use Combinational Pipelining scheduler with the specified pipeline stages")
        ->capture_default_str()
        ->expected(0, 1);
    schedGroup->require_option(0, 1);

    // The callback below is used to set correct optionsCfg.scheduler enum value.
    schedGroup->callback([this, schedGroup] () {
      // Options are mutually exclusive (enforced by require_option(0, 1)),
      // so 3 consecutive if-s should suffice.
      if (!schedGroup->get_option(ASAP_SCHEDULER_FLAG)->empty()) {
        this->optionsCfg.scheduler = dfcxx::Scheduler::ASAP;
      }

      if (!schedGroup->get_option(LP_SCHEDULER_FLAG)->empty()) {
        this->optionsCfg.scheduler = dfcxx::Scheduler::Linear;
      }

      if (!schedGroup->get_option(PIPELINE_SCHEDULER_ARG)->empty()) {
        this->optionsCfg.scheduler = dfcxx::Scheduler::CombPipelining;
	// If this option is used, the parsed number of pipeline stages has
	// already been set to "optionsCfg.stages",
	// so there's no need to do it now.
      }
    });

    auto outputGroup = options->add_option_group(OUTPUT_GROUP);
    outputGroup
        ->add_option(OUT_SV_ARG,
                     outNames[OUT_FORMAT_ID_INT(SystemVerilog)],
                     "Path to output the SystemVerilog module")
        ->default_str(OUT_SV_ARG_DEFAULT)
        ->expected(0, 1);
    outputGroup
         ->add_option(OUT_SV_LIB_ARG,
                      outNames[OUT_FORMAT_ID_INT(SVLibrary)],
                      "Path to output SystemVerilog modules for generated operations")
         ->default_str(OUT_SV_LIB_ARG_DEFAULT)
         ->expected(0, 1);
    outputGroup
         ->add_option(OUT_UNSCHEDULED_DFCIR_ARG,
                      outNames[OUT_FORMAT_ID_INT(UnscheduledDFCIR)],
                      "Path to output unscheduled DFCIR")
         ->default_str(OUT_UNSCHEDULED_DFCIR_ARG_DEFAULT)
         ->expected(0, 1);
    outputGroup
        ->add_option(OUT_SCHEDULED_DFCIR_ARG,
                     outNames[OUT_FORMAT_ID_INT(ScheduledDFCIR)],
                     "Path to output scheduled DFCIR")
        ->default_str(OUT_SCHEDULED_DFCIR_ARG_DEFAULT)
        ->expected(0, 1);
    outputGroup
        ->add_option(OUT_FIRRTL_ARG,
                     outNames[OUT_FORMAT_ID_INT(FIRRTL)],
                     "Path to output scheduled FIRRTL")
        ->default_str(OUT_FIRRTL_ARG_DEFAULT)
        ->expected(0, 1);
    outputGroup
        ->add_option(OUT_DOT_ARG,
                     outNames[OUT_FORMAT_ID_INT(DOT)],
                     "Path to output a DFCxx kernel in DOT format.")
        ->default_str(OUT_DOT_ARG_DEFAULT)
        ->expected(0, 1);
    outputGroup->require_option();
  }

  void fromJson(Json json) override { }
  
  dfcxx::Ops convertFieldToEnum(const std::string field) {
    if (field == "ADD_INT")         { return dfcxx::ADD_INT; }          else
    if (field == "ADD_FLOAT")       { return dfcxx::ADD_FLOAT; }        else
    if (field == "SUB_INT")         { return dfcxx::SUB_INT; }          else
    if (field == "SUB_FLOAT")       { return dfcxx::SUB_FLOAT; }        else
    if (field == "MUL_INT")         { return dfcxx::MUL_INT; }          else
    if (field == "MUL_FLOAT")       { return dfcxx::MUL_FLOAT; }        else
    if (field == "DIV_INT")         { return dfcxx::DIV_INT; }          else
    if (field == "DIV_FLOAT")       { return dfcxx::DIV_FLOAT; }        else
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
    return dfcxx::UNDEFINED;
  }

  void parseInternalOpsConfig(Json json) {
    if (!json.contains(DFCIR_OPS_ID_JSON)) { return; }
    for (auto &[key, val] : json[DFCIR_OPS_ID_JSON].items()) {
      latencyCfg.internalOps[convertFieldToEnum(key)] = val;
    }
  }

  void parseExternalOpsConfig(Json json) {
    if (!json.contains(EXTERNAL_OPS_ID_JSON)) { return; }
    for (auto &[key, val] : json[EXTERNAL_OPS_ID_JSON].items()) {
      latencyCfg.externalOps[key] = val;
    }
  }

  void parseLatencyConfig() {
    if (latencyCfgFile.empty()) { return; }
    std::ifstream in(latencyCfgFile);
    if (!in.good()) { return; }
    Json json = Json::parse(in);
    parseInternalOpsConfig(json);
    parseExternalOpsConfig(json);
  }

  std::string latencyCfgFile;
  dfcxx::DFLatencyConfig latencyCfg;
  std::vector<std::string> outNames;
  bool asapScheduler;
  bool lpScheduler;
  dfcxx::DFOptionsConfig optionsCfg;
};

struct SimOptions final : public AppOptions {

  SimOptions(AppOptions &parent):
      AppOptions(parent, SIM_CMD, "DFCxx simulation") {
    
    // Named options.
    options
        ->add_option(SIM_IN_ARG,
                     inFilePath,
                     "Simulation input data path")
        ->default_str(SIM_IN_ARG_DEFAULT)
        ->expected(0, 1);
    options
        ->add_option(SIM_OUT_ARG,
                     outFilePath,
                     "Simulation results output path")
        ->default_str(SIM_OUT_ARG_DEFAULT)
        ->expected(0, 1);
  }

  void fromJson(Json json) override {
    get(json, SIM_IN_JSON,  inFilePath);
    get(json, SIM_OUT_JSON, outFilePath);
  }

  std::string inFilePath;
  std::string outFilePath;
  std::vector<std::string> files;
};

struct Options final : public AppOptions {
  Options(const std::string &title,
          const std::string &version):
      AppOptions(title, version), hls(*this), sim(*this) {

    // Top-level options.
    options->set_help_all_flag("-H,--help-all", "Print the extended help message and exit");
    options->set_version_flag("-v,--version", version, "Print the tool version");
  }

  void initialize(const std::string &config, int argc, char **argv) {
    // Read the JSON configuration.
    read(config);
    // Command line is of higher priority.
    parse(argc, argv);

    // Subcommand-specific initialization actions.
    if (hls) {
      // Parse latency configuration in case it was supplied.
      hls.parseLatencyConfig();
    }
  }

  int exit(const CLI::Error &e) const {
    return options->exit(e);
  }

  void fromJson(Json json) override {
    hls.fromJson(json[HLS_ID_JSON]);
    sim.fromJson(json[SIM_ID_JSON]);
  }

  HlsOptions hls;
  SimOptions sim;
};
