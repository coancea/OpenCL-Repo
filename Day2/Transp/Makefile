CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function

OS=$(shell uname -s)
ifeq ($(OS),Darwin)
LDFLAGS?=-D DEVICE_ID=1 -framework OpenCL -lm
else
LDFLAGS?=-lOpenCL -lm
endif

PROGRAMS=coalescing

all: $(PROGRAMS)

%: %.c helper.h bridge.h ../../clutils.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
