PYTHON ?= python

.PHONY: smoke train experiments figures matlab-assets paper

smoke:
	$(PYTHON) python/smoke_test.py

train:
	$(PYTHON) python/train_rl.py --mode both --seeds 11 29 47 --updates 24 --episodes-per-update 6

experiments:
	$(PYTHON) python/run_experiments.py

figures:
	MPLCONFIGDIR=.matplotlib $(PYTHON) python/make_figures.py

matlab-assets:
	$(PYTHON) python/export_matlab_assets.py

paper:
	cd paper && latexmk -pdf icbme_rsp_admm_pid.tex
