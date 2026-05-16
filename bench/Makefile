CXX?=g++
# -O3 for benchmark realism. No sanitizers - they distort timings 5-20x.
CXXFLAGS?=-Wall -Wextra -Wpedantic -O3 -std=c++23 -DNDEBUG
INCLUDE=-I../include/
TARGET=bench_main

$(TARGET):bench_main.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	${RM} $(TARGET)

.PHONY: run clean
