CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function
LDFLAGS?=-lOpenCL

PROGRAMS=mmm

all: $(PROGRAMS)

%: %.c mmm.h bridge.h ../../clutils.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
