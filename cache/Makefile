CXX=g++
CXXFLAGS=-Wall -g -ggdb -std=c++14 -larmadillo -lboost_serialization
INCLUDE=-I../include/
TARGET=cache

$(TARGET):cache.cc
			 $(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDE)
clean:
		${RM} $(TARGET1) $(TARGET2)
