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

%option noyywrap
%option nounput

%{
  #include <parser.tab.h>
%}

NUM  [1-9][0-9]+
VAL  0b[0-1]+
VAR  [_a-zA-Z][_a-zA-Z0-9]*
TYPE [i,u]NUM

%%
"input"    { return TOK_INPUT;   }
"output"   { return TOK_OUTPUT;  }
"wire"     { return TOK_WIRE;    }
"reg"      { return TOK_REG;     }
"always"   { return TOK_ALWAYS;  }
"if"       { return TOK_IF;      }
"posedge"  { return TOK_POSEDGE; }
"negedge"  { return TOK_NEGEDGE; }
"level0"   { return TOK_LEVEL0;  }
"level1"   { return TOK_LEVEL1;  }
"*"        { return TOK_STAR;    }
"<="       { return TOK_ASSIGN;  }
"~"        { return TOK_NOT;     }
"&"        { return TOK_AND;     }
"|"        { return TOK_OR;      }
"^"        { return TOK_XOR;     }
"+"        { return TOK_ADD;     }
"-"        { return TOK_SUB;     }
";"        { return TOK_SEMI;    }
"("        { return TOK_LBRACK;  }
")"        { return TOK_RBRACK;  }
"{"        { return TOK_LCURLY;  }
"}"        { return TOK_RCURLY;  }
NUM        {
             yylval = strdup(yytext);
             return TOK_NUM;
           }
VAL        {
             yylval = strdup(yytext);
             return TOK_VAL;
           }
VAR        {
             yylval = strdup(yytext);
             return TOK_VAR;
           }
TYPE       {
             yylval = strdup(yytext);
             return TOK_TYPE;
           }
[ \r\n\t]+ { /* Whitespace */    }
.          { return TOK_OTHER;   }
%%

