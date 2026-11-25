# use multi-stage build, to first build GCTA, then copy the compiled executable
# into a later stage
FROM ubuntu:22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# installed required executables, libraries, and header files
RUN apt update
RUN apt upgrade -y
RUN apt install -y build-essential gpg-agent git cmake curl
RUN apt install -y libboost-all-dev zlib1g-dev libsqlite3-dev libgsl-dev libzstd-dev

# install intel MKL library
RUN curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt update
RUN apt install -y intel-oneapi-mkl-devel

# obtain code repos for appropriate versions
RUN git clone --depth 1 --branch 3.3.7 https://gitlab.com/libeigen/eigen.git
RUN git clone --depth 1 --branch v1.0.0 https://github.com/yixuan/spectra.git

# get latest commit hash from the gcta repository, avoids rebuild if unchanged
ADD https://api.github.com/repos/jeremymcrae/gcta/commits/master /tmp/gcta-head.json
RUN git clone --depth 1 --recurse-submodules https://github.com/jeremymcrae/gcta.git

# set environment variables
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV EIGEN3_INCLUDE_DIR=/eigen
ENV BOOST_LIB=/usr/include/boost
ENV SPECTRA_LIB=/spectra/include

# build GCTA executable
RUN cd gcta && mkdir build && cd build && cmake .. && make VERBOSE=1  

# second stage, copy in executable
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y libgsl27 libzstd1 libgomp1 sqlite3 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /gcta/build/gcta64 /usr/local/bin/gcta
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_gnu_thread.so.3 /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so.3
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.3 /usr/lib/x86_64-linux-gnu/libmkl_core.so.3
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.3 /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so.3
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.3 /usr/lib/x86_64-linux-gnu/libmkl_avx2.so.3
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_def.so.3 /usr/lib/x86_64-linux-gnu/libmkl_vml_def.so.3
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx2.so.3 /usr/lib/x86_64-linux-gnu/libmkl_vml_avx2.so.3
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.3 /usr/lib/x86_64-linux-gnu/libmkl_def.so.3

# cd ~/PATH; podman build --platform linux/amd64 -t gcta -f docker/Dockerfile .
# podman run --platform linux/amd64 gcta:latest gcta -h
# podman save gcta | gzip > ~/gcta.tar.gz
# podman push gcta jeremymcrae/gcta


