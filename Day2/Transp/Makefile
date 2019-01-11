CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function
LDFLAGS?=-lOpenCL

PROGRAMS=coalescing

all: $(PROGRAMS)

%: %.c helper.h bridge.h ../../clutils.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
