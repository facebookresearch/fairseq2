SHELL=bash
VENV=./venv/bin/
ACTIVATE_VENV=source $(VENV)activate
CONDAENV=$$(basename $(CURDIR))
PY_SRC=src/ examples/ tests/

.PHONY: force

fmt:
	$(VENV)black ${PY_SRC}
	$(VENV)isort ${PY_SRC}
	$(VENV)autoflake -ir ${PY_SRC}

check_fmt:
	$(VENV)black --check --diff ${PY_SRC}
	$(VENV)isort --check --diff ${PY_SRC}

pylint:
	$(VENV)flake8 ${PY_SRC}
	$(VENV)mypy ${PY_SRC}

shlint:
	./tools/linters/run-shellcheck.sh src/

lint: pylint shlint

docs:
	cd doc/ && make html SPHINXOPTS="-W"
	cp VERSION doc/build/html

build: venv
	echo "Make sure you have CUDA binaries in your path by running 'module load cuda/11.6'"
	nvcc --version
	$(VENV)pip install -r requirements-build.txt
	git submodule update --init --recursive
	# For Python tools, we can just fetch the executable from VENV,
	# but for cmake, we need to explicitly activate the env.
	$(ACTIVATE_VENV); cmake -GNinja -B build -DFAIRSEQ2_EDITABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
	$(ACTIVATE_VENV); cmake --build build

install: venv build .PHONY
	$(VENV)pip install --upgrade -e .
	$(VENV)pip install --upgrade -r ./requirements-devel.txt
	$(VENV)python -c 'import fairseq2; print(f"fairseq2: {fairseq2.__version__}")'
	# Fairseq2 has been installed in $(VENV)python!

venv:
	python -m venv venv
	$(VENV)pip install -U pip
	$(VENV)pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
	$(VENV)python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print(torch.version.__version__)'
	# Python venv created, run '. ./venv/bin/activate' to activate it
	# You may need to run 'make install' to install fairseq2 inside.

tests: .PHONY
	$(VENV)pytest

integration_tests:
	$(VENV)pytest --integration

integration: check_fmt pylint integration_tests docs

clean:
	[ ! -f build/ ] || rm -r build/
