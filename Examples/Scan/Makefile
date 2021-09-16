CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function
LDFLAGS?=-lOpenCL

PROGRAMS=Scan

all: $(PROGRAMS)

%: %.c ../../clutils.h GenericHack.h Scan.c SetupOpenCL.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
