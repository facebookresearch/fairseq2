# Docker Images for Continuous Integration
The Dockerfiles under this directory serve as the source of the container images
used in GitHub Actions.

## manylinux
The images with the `manylinux` tag are used to build and test fairseq2 on
Linux. They have the tag format `fairseq2-ci-manylinux_<ARCH>:<VERSION>-<VARIANT>`,
where `<ARCH>` is the architecture (e.g. `x86_64`), `<VERSION>` is the current
version of the image (e.g. `1`), and `<VARIANT>` is either `cpu` or a CUDA
version specifier (e.g. `cu117`).

The images are based of PyPA's [manylinux2014](https://github.com/pypa/manylinux)
to ensure maximum binary compatibility with different Linux distributions.

### Deployment Instructions
As of this writing, all images are readily available in the
[ghcr.io/facebookresearch](https://github.com/orgs/facebookresearch/packages)
registry. You should follow these instructions if, for any reason, an image
should be updated. In such case, make sure to increment `<VERSION>` in the
Dockerfile, in GA workflows, and in the commands below.

#### 1. Build the Docker Image
```
docker build --network host --tag ghcr.io/facebookresearch/fairseq2-ci-manylinux_<ARCH>:<VERSION>-<VARIANT> -f Dockerfile.<VARIANT> .
```

#### 2. Push to the GitHub Container Registry
If you don't already have a Personal Access Token with read and write access to
the GitHub Container Registry, follow the steps
[here](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

First, log in to the registry:

```
docker login ghcr.io -u <GITHUB_USERNAME> --password-stdin
```

Then, push the image:

```
docker push ghcr.io/facebookresearch/fairseq2-ci-manylinux_<ARCH>:<VERSION>-<VARIANT>
```

Lastly, log out to avoid any accidental or malicious use of the registry:

```
docker logout ghcr.io
```
