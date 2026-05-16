CXX?=g++
CXXFLAGS?=-Wall -Wextra -g -ggdb -O2 -std=c++23
INCLUDE=-I../include/
LDLIBS=-larmadillo
TARGET=dbscan

$(TARGET):dbscan.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@ $(LDLIBS)
clean:
	${RM} $(TARGET)
