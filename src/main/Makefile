C = ext/aiger/aiger.c

CPP = main.cpp \
      gate/model/gate.cpp \
      gate/model/gsymbol.cpp \
      gate/model/netlist.cpp \
      gate/model/signal.cpp \
      rtl/compiler/compiler.cpp \
      rtl/library/flibrary.cpp \
      rtl/model/event.cpp \
      rtl/model/fsymbol.cpp \
      rtl/model/net.cpp \
      rtl/model/vnode.cpp \
      rtl/model/pnode.cpp \
      rtl/parser/builder.cpp \
      rtl/parser/parser.cpp \
      util/utils.cpp

L = rtl/parser/lexer.l
Y = rtl/parser/parser.y

L_C = lexer.yy.cpp
Y_C = parser.tab.cpp
Y_H = parser.tab.hpp

OBJ = $(C:%.c=%.o) $(CPP:%.cpp=%.o) $(Y_C:%.cpp=%.o) $(L_C:%.cpp=%.o)

EXE = main

CC  = gcc
CXX = g++
LD  = ld

TEST = test.ril

CFLAGS = -I. -g3 -Wall
CXXFLAGS = $(CFLAGS) -std=c++17

all: $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -DYYERROR_VERBOSE -o $@ $^ -lfl

$(Y_C): $(Y)
	bison -d -o $@ $^

$(L_C): $(L)
	flex -o $@ $^

run: $(EXE) $(TEST)
	./$(EXE) $(TEST)

debug: $(EXE)
	gdb --args ./$(EXE)

memcheck: $(EXE)
	valgrind ./$(EXE)

clean:
	rm -rf $(EXE) $(OBJ) $(L_C) $(Y_C) $(Y_H)

