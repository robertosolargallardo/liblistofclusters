# Top-level convenience Makefile.
# Real build is CMake (see CMakeLists.txt). This wrapper exists so
# `make test`, `make examples`, `make all`, `make clean` work from the repo root.

EXAMPLES := cache dbscan baseline_brute_force

.PHONY: all test examples clean $(EXAMPLES)

all: test examples

test:
	$(MAKE) -C tests run

examples: $(EXAMPLES)

$(EXAMPLES):
	$(MAKE) -C examples/$@

clean:
	$(MAKE) -C tests clean
	@for d in $(EXAMPLES); do $(MAKE) -C examples/$$d clean; done
