# Top-level convenience Makefile.
# Real build is CMake (see CMakeLists.txt). This wrapper exists so
# `make test`, `make bench`, `make all`, `make clean` work from the repo root.

.PHONY: all test bench clean

all: test bench

test:
	$(MAKE) -C tests run

bench:
	$(MAKE) -C bench run

clean:
	$(MAKE) -C tests clean
	$(MAKE) -C bench clean
