CXX?=g++
CXXFLAGS?=-Wall -Wextra -Wpedantic -g -O2 -std=c++23 -fsanitize=address,undefined
INCLUDE=-I../include/
TARGET=test_smoke

HEADERS := $(wildcard ../include/listofclusters/*.hh)

$(TARGET):test_smoke.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	${RM} $(TARGET)

.PHONY: run clean
