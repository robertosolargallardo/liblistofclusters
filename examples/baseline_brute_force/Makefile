CXX?=g++
CXXFLAGS?=-Wall -Wextra -g -ggdb -O2 -std=c++23
INCLUDE=-I../../include/
LDLIBS=-larmadillo
TARGET=baseline_brute_force

$(TARGET):all2all.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@ $(LDLIBS)
clean:
	${RM} $(TARGET)
