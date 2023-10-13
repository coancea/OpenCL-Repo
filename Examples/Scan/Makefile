WORKGROUP_SIZE=512
PROGRAMS=Scan

CC?=gcc
CFLAGS?=-std=c99 -O3 -Wall -Wextra -pedantic -Wno-unused-function \
		-DWORKGROUP_SIZE=$(WORKGROUP_SIZE)
LDFLAGS?=-lOpenCL


all: $(PROGRAMS)

%: %.c ../../clutils.h Common.h GenericHack.h Scan.c SetupOpenCL.h
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
