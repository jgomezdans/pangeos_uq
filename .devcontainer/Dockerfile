FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-3

# Set environment variables
ENV PATH /usr/local/conda/bin:$PATH

# Install JupyterLab
RUN conda install -y jupyterlab && conda clean -afy

# Install additional OS-level packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Default shell is bash
SHELL ["/bin/bash", "-c"]

# Make sure conda is initialized
RUN conda init bash

