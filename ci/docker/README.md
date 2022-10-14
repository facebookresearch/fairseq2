# Docker Images for Continuous Integration

The Dockerfiles under this directory serve as the source of our container images
used in GitHub Actions.

All our images have the tag format `fairseq2-ci-wheel:<VERSION>-<VARIANT>`,
where `<VERSION>` is the current version of the image (e.g. `1`) and `<VARIANT>`
is either `cpu` or a CUDA version specifier (e.g. `cu113`).

The images are based of PyPA's
[manylinux2014](https://github.com/pypa/manylinux) to ensure maximum binary
compatibility with different Linux distributions.

## Deployment Instructions
As of this writing, `fairseq2-ci-wheel:1-*` images are readily available in the
ghcr.io/fairinternal registry. You should follow these instructions if, for any
reason, the images should be updated. In such case, make sure to increment
`<VERSION>` both in the Dockerfiles and in your docker commands (see below).

### 1. Build the Docker Image

The `<VARIANT>` must be one of `cpu`, `cu102`, `cu113`, or `cu116`.

```
docker build --network host --tag ghcr.io/fairinternal/fairseq2-ci-wheel:<VERSION>-<VARIANT> -f Dockerfile.<VARIANT> .
```

### 2. Push to the GitHub Container Registry

If you don't already have a Personal Access Token with read and write access to
the GitHub Container Registry, follow the steps
[here](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

First log in to the registry:

```
docker login ghcr.io -u <GITHUB_USERNAME> -p <GITHUB_ACCESS_TOKEN>
```

Then, push the image:

```
docker push ghcr.io/fairinternal/fairseq2-ci-wheel:<VERSION>-<VARIANT>
```

Lastly, log out to avoid any accidental or malicious use of the registry:

```
docker logout ghcr.io
```
