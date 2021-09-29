/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

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
  NODETYPE
  GRAPH
  CHAN
  NODE
  LATENCY
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
  { Builder::get().start_model(); }
  nodetypes
  graphs
  { Builder::get().end_model(); }
;

nodetypes:
  %empty
| nodetypes nodetype
;

nodetype:
  NODETYPE LANGLE LATENCY ASSIGN INT[latency] RANGLE ID[name] {
    Builder::get().start_nodetype(*$name, *$latency);
    delete $latency; delete $name;
  }
  LBRACK args RBRACK ARROW {
    Builder::get().start_outputs();
  }
  LBRACK args RBRACK SEMI {
    Builder::get().end_nodetype();
  }
;

args:
  %empty
| arg
| args COMMA arg
;

arg:
  ID[type] LANGLE REAL[flow] RANGLE ID[name] {
    Builder::get().add_argument(*$name, *$type, *$flow);
    delete $type; delete $flow; delete $name;
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
  // There are special types of nodes: merge*, split*, and delay*.
  NODE ID[type] {
    Builder::get().start_node(*$type);
    delete $type;
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

%%
