.PHONY: all test clean export gate

all:
	$(MAKE) -C engine all

test: all
	$(MAKE) -C engine test
	PYTHONPATH=. pytest tests/

export:
	python -m train.kws.export --ckpt checkpoints/quant_model.pt
	$(MAKE) -C engine clean all

gate: export
	python -m train.kws.eval_c --ckpt checkpoints/quant_model.pt --engine both

clean:
	$(MAKE) -C engine clean
