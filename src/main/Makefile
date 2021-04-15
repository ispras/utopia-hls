SRC = main.cpp \
      gate/gate.cpp \
      rtl/event.cpp \
      rtl/function.cpp \
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
