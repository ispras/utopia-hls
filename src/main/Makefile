SRC = main.cpp \
      gate/gate.cpp \
      gate/gsymbol.cpp \
      gate/signal.cpp \
      rtl/event.cpp \
      rtl/fsymbol.cpp \
      rtl/net.cpp \
      rtl/vnode.cpp \
      rtl/pnode.cpp \
      utils/utils.cpp

EXE = main

CFLAGS = -I. -std=c++17 -g3 -Wall

all: $(EXE)

$(EXE): $(SRC)
	g++ $(CFLAGS) -o $@ $^

run: $(EXE)
	./$(EXE)

debug: $(EXE)
	gdb --args ./$(EXE)

clean:
	rm -rf $(EXE)
