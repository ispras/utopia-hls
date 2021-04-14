SRC = main.cpp net.cpp
EXE = main

CFLAGS = -std=c++17

all: $(EXE)

$(EXE): $(SRC)
	g++ $(CFLAGS) -o $@ $^

clean:
	rm -rf $(EXE)
