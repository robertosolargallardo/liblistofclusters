CXX?=g++
CXXFLAGS?=-Wall -Wextra -Wpedantic -g -O2 -std=c++23 -fsanitize=address,undefined
INCLUDE=-I../include/
TARGET=test_smoke

$(TARGET):test_smoke.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	${RM} $(TARGET)

.PHONY: run clean
