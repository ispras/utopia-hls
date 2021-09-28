C = ext/aiger/aiger.c

CPP = main.cpp \
      gate/model/gate.cpp \
      gate/model/gsymbol.cpp \
      gate/model/netlist.cpp \
      gate/model/signal.cpp \
      hls/model/model.cpp \
      hls/parser/builder.cpp \
      hls/parser/parser.cpp \
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

L = hls/parser/lexer.l \
    rtl/parser/lexer.l

Y = hls/parser/parser.y \
    rtl/parser/parser.y

OBJ = $(C:%.c=%.o) $(CPP:%.cpp=%.o) \
      $(Y:%.y=%.tab.o) $(L:%.l=%.yy.o)

EXE = main

CC  = gcc
CXX = g++
LD  = ld

TEST = ../test/ril/test.ril ../test/hil/test.hil

CFLAGS = -I. -g3 -Wall
CXXFLAGS = $(CFLAGS) -std=c++17

all: $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.tab.cpp: %.y
	bison -d -o $@ $^

%.yy.cpp: %.l
	flex -o $@ $^

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -DYYERROR_VERBOSE -o $@ $^ -lfl

run: $(EXE) $(TEST)
	./$(EXE) $(TEST)

debug: $(EXE)
	gdb --args ./$(EXE)

memcheck: $(EXE)
	valgrind ./$(EXE)

clean:
	rm -rf $(EXE) $(OBJ) $(L:%.l=%.yy.cpp) $(Y:%.y=%.tab.cpp) $(Y:%.y=%.tab.hpp)

