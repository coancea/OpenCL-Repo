CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function
LDFLAGS?=-lOpenCL

PROGRAMS=devices rot13 rot13-profile-simple

all: $(PROGRAMS)

%: %.c clutils.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
