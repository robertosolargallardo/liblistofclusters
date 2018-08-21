CXX=g++
CXXFLAGS=-Wall -g -ggdb -std=c++14 -larmadillo 
INCLUDE=-I../include/
TARGET=dbscan

$(TARGET):dbscan.cc
			 $(CXX) $< -o $@ $(CXXFLAGS) $(INCLUDE)
clean:
		${RM} $(TARGET)
