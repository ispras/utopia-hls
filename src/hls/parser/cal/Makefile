#part of the makefile
parser:	lex_tokens.l parser_cal.y
		bison -d -v --graph parser_cal.y
		flex lex_tokens.l
		cc -o $@ parser_cal.tab.c lex.yy.c -lfl
