# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FROM quay.io/pypa/manylinux2014_x86_64

ENV LIBASAN=/usr/lib64/libasan.so.6
ENV LIBTSAN=/usr/lib64/libtsan.so.0

COPY build-scripts/install-*.sh /build-scripts/

# Install system dependencies.
RUN yum --assumeyes install\
        devtoolset-10-lib{asan,lsan,ubsan,tsan}-devel lib{sndfile,png,jpeg}-devel &&\
    yum clean all

# Install Ninja.
RUN pipx install --pip-args=--no-cache-dir ninja

# Instal LLVM.
RUN /build-scripts/install-llvm.sh

# Install CUDA Toolkits.
RUN /build-scripts/install-cuda-11.6.sh
RUN /build-scripts/install-cuda-11.7.sh
RUN /build-scripts/install-cuda-11.8.sh
RUN /build-scripts/install-cuda-12.1.sh

# Clean up.
RUN rm -rf /build-scripts

COPY manylinux-entrypoint.sh /usr/local/bin/

ENTRYPOINT ["manylinux-entrypoint.sh"]

CMD ["/bin/bash"]
