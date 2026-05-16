CXX?=g++
CXXFLAGS?=-Wall -Wextra -g -ggdb -O2 -std=c++23
INCLUDE=-I../include/
LDLIBS=-larmadillo
TARGET=all2all

$(TARGET):all2all.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@ $(LDLIBS)
clean:
	${RM} $(TARGET)
