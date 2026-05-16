CXX?=g++
# -O3 for benchmark realism. No sanitizers - they distort timings 5-20x.
# hnswlib uses -Wno-unused-parameter style warnings; we suppress just enough
# to compile cleanly without turning -Wall off.
CXXFLAGS?=-Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 -std=c++23 -DNDEBUG -pthread
INCLUDE=-I../include/ -I../third_party/
TARGET=bench_main

# Track header dependencies so changes to the library trigger a rebuild.
HEADERS := $(wildcard ../include/listofclusters/*.hh) $(wildcard ../third_party/hnswlib/*.h)

$(TARGET):bench_main.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@ -pthread

run: $(TARGET)
	./$(TARGET)

clean:
	${RM} $(TARGET)

.PHONY: run clean
