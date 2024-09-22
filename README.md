# Materials for the PANGEOS Summer School 2024
## Radiative transfer inversion and uncertainty propagation

### Introduction

This repository contains a number of codes that demonstrate:
* The basics of radiative transfer (RT) models for leaves and canopies,
* Some experiments to *invert the models*
* The treatment of uncertainty and the role of prior information in inversion.

### Using the materials

#### Installation

You can run the notebooks online using [gitpod](https://gitpod.io/#https://github.com/jgomezdans/pangeos_uq/) without installing anything on your computer.

If you have a github account, you can also fork the repository, and use Github codespaces to run the notebooks on the cloud (using VSCode).

If you want to install and run things locally on your computer, the repository is designed to be used in conjunction with the [conda/mamba Python distribution](). You can follow these steps to install on your laptop (only tested on Linux!)
1. Either
    a) Download [a zip file](https://github.com/jgomezdans/pangeos_uq) file from the repository, and unzip it somewhere.
    b) "Clone" the repository using `git`: `git clone https://github.com/jgomezdans/pangeos_uq`
2. Go to the `pangeos_uq` folder.
3. If you don't have mamba/conda installed, [download the installer](https://mambaforge), and follow the instructions on [here](mambaforge instructions)
4. You can now install all the dependencies using `mamba create env -f environment.yml`
5. After a while, all packages will be installed
6. Activate the environment with `mamba activate pangeos_uq`
7. Change directory to `notebooks`, and launch `jupyter lab`
8. Use notebooks on the browser

### Additional software
The notebooks make use of codes defined in the `pangeos_uq` Python package. These are found in the `src/` folder in the main repository. Feel free to have a look and poke around. Improvements and bugfixes are welcomed, make sure you let me know by flagging a [Github Issue report]().