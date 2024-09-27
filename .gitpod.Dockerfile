# Dockerfile

FROM condaforge/mambaforge

# Copy your environment.yml into the image
COPY environment.yml /tmp/environment.yml

# Create the Conda environment
RUN conda env create -f /tmp/environment.yml && conda clean --all --yes && conda init bash

# Set Conda to automatically activate the environment
RUN conda activate pangeos_uq

RUN echo 'create-overlay $HOME /lib ' > "$HOME/.runonce/1-home-lib_persist"


# Create the notebooks directory in the workspace
RUN mkdir -p /workspace/
