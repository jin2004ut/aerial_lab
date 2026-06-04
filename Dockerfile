# =============================================================================
# Aerial Lab - IsaacLab RL Environment for Articulated Aerial Robots
# Base: CUDA 12.8 + Ubuntu 22.04
#
# Requirements on host:
#   - NVIDIA driver >= 570
#   - NVIDIA Container Toolkit (nvidia-container-toolkit)
#   - Docker >= 24
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Build arguments
ARG ISAACLAB_COMMIT=47780cf02dae94410cfed81706c8c859eeeacd76
ARG CONDA_ENV=aeriallab
ARG PYTHON_VERSION=3.11

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    cmake \
    build-essential \
    liburdfdom-tools \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libvulkan1 \
    vulkan-tools \
    libegl1-mesa-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install Miniforge (conda-forge based, no Anaconda ToS required)
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -a -y

ENV PATH="${CONDA_DIR}/bin:$PATH"

# Initialize conda for bash
RUN conda init bash

# Create the aeriallab conda environment
RUN conda create -n ${CONDA_ENV} python=${PYTHON_VERSION} -y && conda clean -a -y

# Convenience shorthand – use this env's pip/python for all subsequent installs
ENV PIP="${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip"
ENV PYTHON="${CONDA_DIR}/envs/${CONDA_ENV}/bin/python"

# Keep setuptools on a version that still ships pkg_resources.
# Some Isaac Lab transitive deps (for example flatdict) still import it in setup.py.
RUN printf 'setuptools<81\n' > /tmp/pip-constraints.txt
ENV PIP_CONSTRAINT=/tmp/pip-constraints.txt

# ---------------------------------------------------------------------------
# Python dependencies: PyTorch (CUDA 12.8)
# ---------------------------------------------------------------------------
RUN ${PIP} install --upgrade pip wheel "setuptools<81" && \
    ${PIP} install \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Isaac Sim 5.0.0  (large download – ~30 GB)
# ---------------------------------------------------------------------------
RUN ${PIP} install \
    "isaacsim[all,extscache]==5.0.0" \
    --extra-index-url https://pypi.nvidia.com

# ---------------------------------------------------------------------------
# Isaac Lab (specific commit for compatibility)
# ---------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout ${ISAACLAB_COMMIT}

# Install Isaac Lab core packages in a deterministic order.
# Using an explicit loop makes failures visible and avoids xargs exit code 123.
WORKDIR /workspace/IsaacLab
RUN set -eux; \
    for pkg in source/isaaclab source/isaaclab_assets source/isaaclab_tasks source/isaaclab_rl; do \
        if [ -d "$pkg" ]; then \
            ${PIP} install -e "$pkg"; \
        fi; \
    done; \
    if [ -d source/isaaclab_mimic ]; then \
        ${PIP} install -e source/isaaclab_mimic; \
    fi

# Explicitly add all IsaacLab source roots to PYTHONPATH so that even if
# editable installs are not picked up at runtime, the packages are importable.
ENV PYTHONPATH="/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_mimic:${PYTHONPATH}"

# Install RL libraries (equivalent to: ./isaaclab.sh --install rl_games rsl_rl sb3 skrl robomimic)
RUN ${PIP} install \
    rl-games \
    rsl-rl-lib==3.0.1 \
    stable-baselines3 \
    skrl \
    robomimic

# ---------------------------------------------------------------------------
# Aerial Lab project
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY . /workspace/aerial_lab

WORKDIR /workspace/aerial_lab
RUN ${PIP} install -e source/aerial_lab

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
# Make the aeriallab conda env the default for interactive shells
RUN echo "conda activate ${CONDA_ENV}" >> /root/.bashrc

# Put the conda env on PATH so scripts run without explicit activation
ENV PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:$PATH"
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

# Isaac Sim headless / EGL rendering
ENV ISAACSIM_HEADLESS=1
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/aerial_lab
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
