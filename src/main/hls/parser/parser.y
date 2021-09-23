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

  #include "hls/parser/builder.h"

  using namespace eda::hls;
  using namespace eda::hls::parser;

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
  GRAPH
  CHAN
  NODE
  MERGE
  SPLIT
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
;

nodetypes:
  %empty
| nodetypes nodetype
;

nodetype:
  NODE LANGLE LATENCY ASSIGN INT[latency] RANGLE ID[name]
      LBRACK args RBRACK ARROW
      LBRACK args RBRACK SEMI {
    delete $latency; delete $name;
  }
;

args:
  %empty
| arg
| args COMMA arg
;

arg:
  ID[type] LANGLE REAL[flow] RANGLE ID[name]
;

graphs:
  %empty
| graphs graph
;

graph:
  GRAPH ID[name] LCURLY
    chans
    nodes
  RCURLY
;

chans:
  %empty
| chans chan
;

chan:
  CHAN ID[type] ID[name] SEMI

nodes:
  %empty
| nodes node
;

node:
  NODE ID[name] LBRACK params RBRACK ARROW
    LBRACK params RBRACK SEMI
| NODE MERGE LBRACK params RBRACK ARROW
    LBRACK params RBRACK SEMI
| NODE SPLIT LBRACK params RBRACK ARROW
    LBRACK params RBRACK SEMI
;

params:
  %empty
| param
| params COMMA param
;

param:
  ID
;

%%
