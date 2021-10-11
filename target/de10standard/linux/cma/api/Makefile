PATH_SETTINGS?=$(PWD)/../Settings.mak
# Driver configuration file
include $(PATH_SETTINGS)

DEFINES=-DCMA_DEBUG=$(CMA_DEBUG) \
		-DDRIVER_NODE_NAME="\"$(DRIVER_NODE_NAME)\""

CC=gcc
LIBRARY_NAME=libcma.a
INC=-I../driver -I../include
CFLAGS=-Wall -O3
ARFLAGS=rcs
OBJ=obj/cma_api.o



all: $(OBJ)
	$(CROSS_COMPILE)$(AR) $(ARFLAGS) $(LIBRARY_NAME) $(OBJ)

obj/%.o:src/%.c
	$(CROSS_COMPILE)$(CC) $(CFLAGS) $(DEFINES) $(INC) -c $< -o $@

clean:
	$(RM) $(LIBRARY_NAME)
	$(RM) obj/*
