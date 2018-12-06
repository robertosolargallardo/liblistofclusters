CXX=g++
CXXFLAGS=-Wall -g -ggdb -std=c++14 -larmadillo 
INCLUDE=-I../include/
TARGET=all2all

$(TARGET):all2all.cc
			 $(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDE)
clean:
		${RM} $(TARGET)
