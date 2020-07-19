CXX        = nvcc -O3

SRC1 = transpose-main.cu
SRC2 = mmm-main.cu
SOURCES_CPP = transpose-main.cu mmm-main.cu
HELPERS1 = transpose-host.cu.h transpose-kernels.cu.h
HELPERS2 = mmm-kernels.cu.h
EXECUTABLE1 =transp
EXECUTABLE2 =mmm

default: compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE1) $(EXECUTABLE2)

$(EXECUTABLE1): $(SRC1) $(HELPERS1)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)

$(EXECUTABLE2): $(SRC2) $(HELPERS2)
	$(CXX) -o $(EXECUTABLE2) $(SRC2)

run: $(EXECUTABLE1) $(EXECUTABLE2)
	./$(EXECUTABLE2)
#	./$(EXECUTABLE1)

clean:
	rm -f $(EXECUTABLE1) $(EXECUTABLE2)

