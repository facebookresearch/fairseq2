SHELL=bash
CONDA_PREFIX?=./venv
VENV?=$(CONDA_PREFIX)/bin/
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
	./ci/scripts/run-shellcheck.sh src/

cpplint:
	run-clang-tidy -p build -config="{InheritParentConfig: true, WarningsAsErrors: '*'}" -quiet -extra-arg='-std=c++17' -fix

lint: pylint shlint cpplint

doc: .PHONY
	rm -r doc/build || true
	rm -r doc/reference/generated || true
	$(VENV)sphinx-build doc/ doc/build/html/ -aW --keep-going || ( \
		echo 'List of sphinx references:'; \
		$(VENV)python -c 'import pickle; [print(r) for r in pickle.load(open("doc/build/html/.doctrees/environment.pickle","rb")).domaindata["std"]["labels"].keys()]'; false \
	)
	cp VERSION doc/build/html
	# firefox doc/build/html/index.html

build: deps build/src/fairseq2/native/libfairseq2.so

deps:
	$(VENV)pip install -r requirements-build.txt
	git submodule update --init --recursive

build/src/fairseq2/native/libfairseq2.so: .PHONY
	$(VENV)cmake -GNinja -B build \
		-DCMAKE_PREFIX_PATH=$(VENV).. \
		-DFAIRSEQ2_EDITABLE_PYTHON=ON \
		-DFAIRSEQ2_BUILD_FOR_NATIVE=ON \
		-DFAIRSEQ2_TREAT_WARNINGS_AS_ERRORS=ON \
		-DFAIRSEQ2_PERFORM_LTO=ON \
		-DCMAKE_BUILD_TYPE=Release

	$(VENV)cmake --build build

install: venv build .PHONY
	$(VENV)pip install --upgrade -e .
	$(VENV)pip install --upgrade -r ./requirements-devel.txt
	$(VENV)pip install --upgrade -r ./doc/requirements.txt
	$(VENV)python -c 'import fairseq2; print(f"fairseq2: {fairseq2.__version__}")'
	# Fairseq2 has been installed in $(VENV)python!

venv: $(VENV)

$(VENV):
	python3 -m venv venv
	$(VENV)pip install -U pip
	$(VENV)pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
	$(VENV)python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print(f"torch=={torch.__version__}")'
	# Python venv created, run '. ./venv/bin/activate' to activate it
	# You may need to run 'make install' to install fairseq2 inside.

tests: .PHONY
	$(VENV)pytest

integration_tests:
	$(VENV)pytest --integration

integration: check_fmt pylint integration_tests docs

clean:
	[ ! -e build/ ] || rm -r build/

optview: .PHONY
	# Shows missed optimization on our code
	CC=clang CXX=/usr/bin/clang++ CXXFLAGS='-fsave-optimization-record' make clean build
	$(VENV)python $(HOME)/github/optview2/opt-viewer.py build/src/fairseq2/native/CMakeFiles/fairseq2.dir/ --source-dir src/
	firefox ./html/index.html
