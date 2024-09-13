# Image source code: https://github.com/axonasif/workspace-images/tree/tmp
# Also see https://github.com/gitpod-io/workspace-images/issues/1071
FROM gitpod/workspace-base

# Set user
USER gitpod

# Install Miniconda
RUN wget --quiet \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh

# Add Miniconda to PATH
ENV PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda in bash config files:
RUN conda init bash

# Set up Conda channels
RUN conda config --add channels jgomezdans && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Set libmamba as solver
RUN conda config --set solver libmamba

# Persist ~/ (HOME) and lib
RUN echo 'create-overlay $HOME /lib' > "$HOME/.runonce/1-home-lib_persist"

# Create an alias
RUN echo 'alias gogo="$GITPOD_REPO_ROOT/utils/gogo.sh"' >> $HOME/.bash_aliases
RUN echo 'alias nb_load="$GITPOD_REPO_ROOT/utils/nb_load.sh"' >> $HOME/.bash_aliases

# Referenced in `.vscode/settings.json`
ENV PYTHON_INTERPRETER="$HOME/miniconda/bin/python"
# Pycharm recognizes this variables
ENV PYCHARM_PYTHON_PATH="${PYTHON_INTERPRETER}"