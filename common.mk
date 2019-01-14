# Generic Makefile definitions.  Import from a Makefile, and be
# careful if changing it here.

CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function

OS=$(shell uname -s)
ifeq ($(OS),Darwin)
LDFLAGS?=-framework OpenCL -lm
else
LDFLAGS?=-lOpenCL -lm
endif

INCLUDES?=

all: $(PROGRAMS)

%: %.c $(INCLUDES)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

run: $(PROGRAMS)
	@for prog in $(PROGRAMS); do echo "./$$prog"; ./$$prog; done

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
