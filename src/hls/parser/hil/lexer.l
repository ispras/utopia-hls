/*===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===*/

%option prefix="hh"
%option yylineno
%option noyywrap
%option nounput

%{
  #include <string>
  #include <hls/parser/hil/parser.tab.hpp>
%}

REAL [0-9]+.[0-9]+
INT  [0-9]+
ID   [_a-zA-Z][_a-zA-Z0-9]*

%%
"nodetype" { return TOK_NODETYPE; }
"graph"    { return TOK_GRAPH;    }
"chan"     { return TOK_CHAN;     }
"node"     { return TOK_NODE;     }
"#"        { return TOK_SHARP;    }
"="        { return TOK_ASSIGN;   }
"=>"       { return TOK_ARROW;    }
","        { return TOK_COMMA;    }
";"        { return TOK_SEMI;     }
"("        { return TOK_LBRACK;   }
")"        { return TOK_RBRACK;   }
"<"        { return TOK_LANGLE;   }
">"        { return TOK_RANGLE;   }
"{"        { return TOK_LCURLY;   }
"}"        { return TOK_RCURLY;   }
{INT}      {
             hhlval = new std::string(yytext);
             return TOK_INT;
           }
{REAL}     {
             hhlval = new std::string(yytext);
             return TOK_REAL;
           }
{ID}       {
             hhlval = new std::string(yytext);
             return TOK_ID;
           }
[ \t\r\n]+ { /* Whitespaces */   }
.          { return TOK_OTHER;   }
%%

