%{
#include "parser_cal.tab.h"
%}

%%
"call"      { return Key_call; }
"at"        { return Key_at; }
"at*"       { return Key_at2; }
"else"      { return Key_else; }
"end"       { return Key_end; }
"if"        { return Key_if; }
"while"     { return Key_while; }
"foreach"   { return Key_foreach; }
"actor"     { return Key_actor; }
"endactor"  { return Key_endactor; }
"endif"     { return Key_endif; }
"action"    { return Key_action; }
"multi"     { return Key_multi; }
"time"      { return Key_time; }
"import"    { return Key_import; }
"all"       { return Key_all; }
"mutable"   { return Key_mutable; }
"old"       { return Key_old; }
"true"      { return Key_true; }
"false"     { return Key_false; }
"null"      { return Key_null; }
"then"      { return Key_then; }
"let"       { return Key_let; }
"endlet"    { return Key_endlet; }
"const"     { return Key_const; }
"lambda"    { return Key_lambda; }
"endlambda" { return Key_endlambda; }
"var"       { return Key_var; }
"proc"      { return Key_proc; }
"endproc"   { return Key_endproc; }
"do"        { return Key_do; }
"begin"     { return Key_begin; }
"function"  { return Key_function; }
"procedure" { return Key_procedure; }
"map"       { return Key_map; }
"for"       { return Key_for; }
"in"        { return Key_in; }
"endwhile"  { return Key_endwhile; }
"endforeach" { return Key_endforeach; }
"choose"    { return Key_choose; }
"endchoose" { return Key_endchoose; }
"endaction" { return Key_endaction; }
"guard"     { return Key_guard; }
"delay"     { return Key_delay; }
"any"       { return Key_any; }
"repeat"    { return Key_repeat; }
"initialize" { return Key_initialize; }
"endinitialize" { return Key_endinitialize; }
"schedule"   { return Key_schedule; }
"endschedule" { return Key_endschedule; }
"fsm"       { return Key_fsm; }
"regexp"    { return Key_regexp; }
"priority"  { return Key_priority; }
[a-zA-Z_$][a-zA-Z0-9_]*  { return Identifiers; }
"<"     { return Oper_less; }
"="     { return Oper_equal; }
"==>"   { return Oper_eqarrow; }
"-->"   { return Oper_doublearrow; }
":="    { return Oper_assignment; }
"|"     { return Oper_vert; }
"->"    { return Oper_arrow; }
"*"     { return Oper_star; }
">"     { return Oper_more; }
"+"     { return Oper_plus; }
"-"     { return Oper_minus; }
"/"     { return Oper_div; }
"."     { return Delimiters_point; }
","     { return Delimiters_comma; }
"("     { return Delimiters_cir_op; }
")"     { return Delimiters_cir_cl; }
"["     { return Delimiters_sq_op; }
"]"     { return Delimiters_sq_cl; }
"{"     { return Delimiters_fig_op; }
"}"     { return Delimiters_fig_cl; }
":"     { return Delimiters_colon; }
";"     { return Delimiters_semicolon; }
[0-9]+  {yylval = atoi(yytext); return Numeric_literals; }
[ \t]*  {}
"//".*\n   {}
"/*".*"*/"   {}
\n      {}
.       { printf("Mystery character %c\n", *yytext); }
%%
