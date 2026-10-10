.PHONY: all test clean

all:
	$(MAKE) -C engine all

test: all
	$(MAKE) -C engine test
	PYTHONPATH=. pytest tests/

clean:
	$(MAKE) -C engine clean
