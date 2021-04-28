SRC = main.cpp \
      gate/flibrary.cpp \
      gate/gate.cpp \
      gate/gsymbol.cpp \
      gate/netlist.cpp \
      gate/signal.cpp \
      rtl/event.cpp \
      rtl/fsymbol.cpp \
      rtl/net.cpp \
      rtl/vnode.cpp \
      rtl/pnode.cpp \
      rtl/parser/builder.cpp \
      utils/utils.cpp

L = rtl/parser/lexer.l
Y = rtl/parser/parser.y

L_C = lexer.yy.c
Y_C = parser.tab.c
Y_H = parser.tab.h

EXE = main

CFLAGS = -I. -std=c++17 -g3 -Wall -DYYERROR_VERBOSE

all: $(EXE)

$(EXE): $(SRC) $(L_C) $(Y_C)
	g++ $(CFLAGS) -o $@ $^ -lfl

$(L_C): $(L)
	flex -o $@ $^

$(Y_C): $(Y)
	bison -d $^

run: $(EXE)
	./$(EXE)

debug: $(EXE)
	gdb --args ./$(EXE)

clean:
	rm -rf $(EXE) $(L_C) $(Y_C) $(Y_H)

