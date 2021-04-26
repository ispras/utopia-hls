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
  int yylex(void);
  void yyerror(const char *) {}
%}

%define api.value.type {char*}
%define api.token.prefix {TOK_}
%token
  INPUT
  OUTPUT
  WIRE
  REG
  ALWAYS
  IF
  POSEDGE
  NEGEDGE
  LEVEL0
  LEVEL1
  STAR
  ASSIGN
  NOT
  AND
  OR
  XOR
  ADD
  SUB
  SEMI
  LBRACK
  RBRACK
  LCURLY
  RCURLY
  NUM
  VAL
  VAR
  TYPE
  OTHER
;

%%

model:
    %empty
  | model item
;

item:
    decl
  | proc
;

decl:
    INPUT  TYPE VAR SEMI
  | OUTPUT TYPE VAR SEMI
  | WIRE   TYPE VAR SEMI
  | REG    TYPE VAR SEMI
;

proc:
    ALWAYS LBRACK event RBRACK
      guard LCURLY action RCURLY
;

event:
    POSEDGE LBRACK VAR RBRACK
  | NEGEDGE LBRACK VAR RBRACK
  | LEVEL0  LBRACK VAR RBRACK
  | LEVEL1  LBRACK VAR RBRACK
  | STAR
;

guard:
    %empty
  | IF LBRACK VAR RBRACK
;

action:
    %empty
  | action assign
;

assign:
    VAR ASSIGN VAL         SEMI
  | VAR ASSIGN VAR         SEMI
  | VAR ASSIGN NOT VAR     SEMI
  | VAR ASSIGN VAR AND VAR SEMI
  | VAR ASSIGN VAR OR  VAR SEMI
  | VAR ASSIGN VAR XOR VAR SEMI
  | VAR ASSIGN VAR ADD VAR SEMI
  | VAR ASSIGN VAR SUB VAR SEMI
;

%%
