# fairseq2 documents

## Install dependencies

Follow the installation instructions and install fairseq2 and fairseq2n.

## Build the docs

```bash
# Install dependencies.
pip install -r requirements.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d build/html/
```
Launch your browser and open localhost:8000.


## Run doctest

```bash
python -m pytest doc/source --doctest-glob=*.rst
```
