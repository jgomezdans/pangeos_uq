# Dockerfile

FROM gitpod/workspace-full

# Install Miniforge in the user's home directory
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p /home/gitpod/miniforge3 \
    && rm Miniforge3-Linux-x86_64.sh \
    && /home/gitpod/miniforge3/bin/conda init bash

# Add Miniforge to the PATH
ENV PATH="/home/gitpod/miniforge3/bin:$PATH"

# Copy your environment.yml into the image
COPY environment.yml /workspace/environment.yml

# Create the Conda environment
RUN conda env create -f /workspace/environment.yml && conda clean --all --yes

# Set Conda to automatically activate the environment
SHELL ["bash", "-c", "source /home/gitpod/miniforge3/bin/activate pangeos_uq && conda activate pangeos_uq"]

RUN echo 'create-overlay $HOME /lib' > "$HOME/.runonce/1-home-lib_persist"


# Create the notebooks directory in the workspace
RUN mkdir -p /workspace/notebooks

# Expose port 8888 for Jupyter Lab
EXPOSE 8888

# Start Jupyter Lab when the container starts in the notebooks folder
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/workspace/notebooks"]
