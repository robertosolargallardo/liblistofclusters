CXX?=g++
CXXFLAGS?=-Wall -Wextra -g -ggdb -O2 -std=c++23
INCLUDE=-I../include/
LDLIBS=-larmadillo
TARGET=cache

$(TARGET):cache.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@ $(LDLIBS)
clean:
	${RM} $(TARGET)
