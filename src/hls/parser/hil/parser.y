//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

%{
  #include <string>
  #include <iostream>

  #include "hls/parser/hil/builder.h"

  using namespace eda::hls;
  using namespace eda::hls::parser::hil;

  extern int hhlineno;
  extern char* hhtext;

  extern int hhlex(void);

  void hherror(const char *error) {
    std::cerr << "Error(" << hhlineno << "): " << error << std::endl;
    std::cerr << hhtext << std::endl;
  }
%}

%define api.prefix {hh}
%define api.value.type {std::string*}
%define api.token.prefix {TOK_}
%token
  MODEL
  NODETYPE
  GRAPH
  CHAN
  NODE
  SHARP
  ASSIGN
  ARROW
  COMMA
  SEMI
  LBRACK
  RBRACK
  LANGLE
  RANGLE
  LCURLY
  RCURLY
  INT
  REAL
  ID
  OTHER
;

%%

model:
  MODEL ID[name] LCURLY {
    Builder::get().start_model(*$name);
    delete $name;
  }
  nodetypes
  graphs
  RCURLY {
    Builder::get().end_model();
  }
;

nodetypes:
  %empty
| nodetypes nodetype
;

nodetype:
  NODETYPE ID[name] {
    Builder::get().start_nodetype(*$name);
    delete $name;
  }
  LBRACK ports RBRACK ARROW {
    Builder::get().start_outputs();
  }
  LBRACK ports RBRACK SEMI {
    Builder::get().end_nodetype();
  }
;

ports:
  %empty
| port
| ports COMMA port
;

port:
  ID[type] LANGLE REAL[flow] RANGLE ID[name] {
    Builder::get().add_port(*$name, *$type, *$flow, "0");
    delete $type; delete $flow; delete $name;
  }
| ID[type] LANGLE REAL[flow] RANGLE SHARP INT[latency] ID[name] {
    Builder::get().add_port(*$name, *$type, *$flow, *$latency);
    delete $type; delete $flow; delete $latency; delete $name;
  }
| ID[type] LANGLE REAL[flow] RANGLE ID[name] ASSIGN INT[value] {
    Builder::get().add_port(*$name, *$type, *$flow, "0", *$value);
    delete $type; delete $flow; delete $name; delete $value;
  }
| ID[type] LANGLE REAL[flow] RANGLE SHARP INT[latency] ID[name] ASSIGN INT[value] {
    Builder::get().add_port(*$name, *$type, *$flow, *$latency, *$value);
    delete $type; delete $flow; delete $latency; delete $name; delete $value;
  }
;

graphs:
  %empty
| graphs graph
;

graph:
  GRAPH ID[name] LCURLY {
    Builder::get().start_graph(*$name);
    delete $name;
  }
  chans
  nodes
  insts
  RCURLY {
    Builder::get().end_graph();
  }
;

chans:
  %empty
| chans chan
;

chan:
  CHAN ID[type] ID[name] SEMI {
    Builder::get().add_chan(*$type, *$name);
    delete $type; delete $name;
  }

nodes:
  %empty
| nodes node
;

node:
  // There are special types of nodes: merge*, split*, dup*, and delay*.
  NODE ID[type] ID[name] {
    Builder::get().start_node(*$type, *$name);
    delete $type; delete $name;
  }
  LBRACK params RBRACK ARROW {
    Builder::get().start_outputs();
  }
  LBRACK params RBRACK SEMI {
    Builder::get().end_node();
  }
;

params:
  %empty
| param
| params COMMA param
;

param:
  ID[name] {
    Builder::get().add_param(*$name);
    delete $name;
  }
;

insts:
  %empty
| insts inst
;

inst:
  GRAPH ID[type] ID[name] {
    Builder::get().start_inst(*$type, *$name);
    delete $type; delete $name;
  }
  LBRACK binds RBRACK ARROW {
    Builder::get().start_outputs();
  }
  LBRACK binds RBRACK SEMI {
    Builder::get().end_inst();
  }
;

binds:
  %empty
| bind
| binds COMMA bind
;

bind:
  ID[name] {
    Builder::get().start_bind(*$name);
    delete $name;
  }
  LBRACK params RBRACK {
    Builder::get().end_bind();
  }
;

%%
