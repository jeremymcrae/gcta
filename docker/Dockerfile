# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=24.04

# -----------------------------------------------------------------------------
# Stage 1: Build GCTA from the local repository checkout
# Ubuntu 24.04 LTS (GCC 13.2) is used as the default rather than 26.04 (GCC 15.2)
# because Eigen 3.3.x predates the EIGEN_GCC15_SLP_BUG workaround for GCC 15.1/15.2
# (GCC PR tree-optimization/124802).
# -----------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG EIGEN_VERSION=3.3.7
ARG SPECTRA_VERSION=1.0.0

# Install compiler toolchain, minimal Boost headers, and static C library archives.
# On amd64, install Intel oneAPI MKL core-devel (skips SYCL GPU & MPI cluster bloat);
# on arm64, install OpenBLAS.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gpg \
        libboost-dev \
        libgsl-dev \
        libsqlite3-dev \
        libzstd-dev \
        zlib1g-dev && \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        apt-get install -y --no-install-recommends libopenblas-dev; \
    else \
        curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
            | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
            > /etc/apt/sources.list.d/oneAPI.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends intel-oneapi-mkl-core-devel; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Download Eigen 3.3.7 and Spectra 1.0.0 release tarballs (cached across source edits)
RUN mkdir -p /opt/deps/eigen /opt/deps/spectra && \
    curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" \
        | tar -xz --strip-components=1 -C /opt/deps/eigen && \
    curl -fsSL "https://github.com/yixuan/spectra/archive/refs/tags/v${SPECTRA_VERSION}.tar.gz" \
        | tar -xz --strip-components=1 -C /opt/deps/spectra

ENV MKLROOT=/opt/intel/oneapi/mkl/latest \
    OPENBLAS=/usr \
    EIGEN3_INCLUDE_DIR=/opt/deps/eigen \
    BOOST_LIB=/usr/include \
    SPECTRA_LIB=/opt/deps/spectra/include

WORKDIR /src/gcta
COPY . .

# Statically link MKL (on amd64) + gsl/sqlite3/zstd/zlib with --gc-sections and strip
RUN if [ ! -f submods/plink-ng/2.0/plink2_base.cc ]; then \
        rm -rf submods/plink-ng && \
        git clone --depth 1 https://github.com/chrchang/plink-ng.git submods/plink-ng; \
    fi && \
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTATIC_DEPS=ON \
        -DBUILD_TESTING=OFF \
        -DARCH_FLAGS="-march=armv8-a" && \
    cmake --build build -j"$(nproc)" && \
    strip --strip-unneeded build/gcta64 && \
    MULTIARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH)" && \
    mkdir -p "/runtime-root/usr/lib/${MULTIARCH}" && \
    cp -a "/usr/lib/${MULTIARCH}/libgomp.so.1"* "/runtime-root/usr/lib/${MULTIARCH}/"

# -----------------------------------------------------------------------------
# Optional ultra-minimal shell-less target (--target distroless, ~70 MB on amd64)
# -----------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian12:nonroot AS distroless
COPY --from=builder /runtime-root/ /
COPY --from=builder /src/gcta/build/gcta64 /usr/local/bin/gcta
ENTRYPOINT ["/usr/local/bin/gcta"]

# -----------------------------------------------------------------------------
# Stage 2 (default): Minimal Ubuntu 24.04 LTS runtime (~130 MB vs ~670 MB before)
# Keeps /bin/sh & /bin/bash for Nextflow, Snakemake, and WDL/Cromwell pipelines.
# -----------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH

RUN apt-get update && \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        apt-get install -y --no-install-recommends libgomp1 libopenblas0; \
    else \
        apt-get install -y --no-install-recommends libgomp1; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/gcta/build/gcta64 /usr/local/bin/gcta
RUN ln -s /usr/local/bin/gcta /usr/local/bin/gcta64

# Build amd64 image (static Intel MKL, ~130 MB uncompressed / ~48 MB .tar.gz):
#   podman build --platform linux/amd64 -t gcta -f docker/Dockerfile .
# Build native arm64 image (Apple Silicon / AWS Graviton):
#   podman build --platform linux/arm64 -t gcta:arm64 -f docker/Dockerfile .
# Optional shell-less distroless image (~70 MB uncompressed / ~28 MB .tar.gz, amd64):
#   podman build --platform linux/amd64 --target distroless -t gcta:slim -f docker/Dockerfile .
# Run & export:
#   podman run --rm --platform linux/amd64 gcta:latest gcta -h
#   podman save gcta | gzip > ~/gcta.tar.gz
#   podman push gcta jeremymcrae/gcta
