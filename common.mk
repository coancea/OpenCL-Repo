# Generic Makefile definitions.  Import from a Makefile, and be
# careful if changing it here.

CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function

OS=$(shell uname -s)
ifeq ($(OS),Darwin)
LDFLAGS?=-framework OpenCL
else
LDFLAGS?=-lOpenCL
endif



all: $(PROGRAMS)

%: %.c ../clutils.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
