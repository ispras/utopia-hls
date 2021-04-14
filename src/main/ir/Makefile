SRC = main.cpp net.cpp vnode.cpp pnode.cpp event.cpp function.cpp utils.cpp
EXE = main

CFLAGS = -std=c++17 -g3

all: $(EXE)

$(EXE): $(SRC)
	g++ $(CFLAGS) -o $@ $^

run: $(EXE)
	./$(EXE)

debug: $(EXE)
	gdb --args ./$(EXE)

clean:
	rm -rf $(EXE)
