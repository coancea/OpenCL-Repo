CXX        = nvcc -O3

SRC1 = Histo-CUB.cu
HELPERS1 = histo-helper.cu.h
EXECUTABLE1 = histo-cub

SRC2 = RedByKey-CUB.cu
HELPERS2 = redbykey-helper.cu.h
EXECUTABLE2 = redbykey-cub

default: compile

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE1) $(EXECUTABLE2)

$(EXECUTABLE1): $(SRC1) $(HELPERS1)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)

$(EXECUTABLE2): $(SRC2) $(HELPERS2)
	$(CXX) -o $(EXECUTABLE2) $(SRC2)

clean:
	rm -f $(EXECUTABLE1) $(EXECUTABLE2)

