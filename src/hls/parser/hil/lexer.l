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

%option prefix="hh"
%option yylineno
%option noyywrap
%option nounput

%{
  #include <string>
  #include <hls/parser/hil/parser.tab.hpp>
%}

REAL [0-9]+(.[0-9]+)?
INT  [1-9][0-9]*
ID   [_a-zA-Z][_a-zA-Z0-9]*

%%
"nodetype" { return TOK_NODETYPE; }
"graph"    { return TOK_GRAPH;    }
"chan"     { return TOK_CHAN;     }
"node"     { return TOK_NODE;     }
"latency"  { return TOK_LATENCY;  }
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

