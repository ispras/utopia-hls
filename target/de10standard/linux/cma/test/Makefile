CC=gcc
CROSS_COMPILE=arm-linux-gnueabihf-
INCLUDES=-I../include/
LIBRARIES=-L../api/ \
		  -lcma \
		  -lrt
EXECUTABLE=cma_test.elf
OBJ=obj/main.o \
	obj/timer.o


all: $(OBJ)
	$(CROSS_COMPILE)$(CC) $(LFLAGS) $(OBJ) -o $(EXECUTABLE) $(LIBRARIES)

obj/%.o:src/%.c
	$(CROSS_COMPILE)$(CC) $(INCLUDES) -c $< -o $@

obj/%.o:src/timer/%.c
	$(CROSS_COMPILE)$(CC) $(INCLUDES) -c $< -o $@


clean:
	$(RM) $(OBJ)
	$(RM) $(EXECUTABLE)
