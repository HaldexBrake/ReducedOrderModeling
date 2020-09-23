# ReducedOrderModeling

This repository contains the implementation of the two methods for reduced order modelling (ROM) considered in the master's thesis [Reduced Order Modelling using Dynamic Mode Decomposition and Koopman Spectral Analysis with Deep Learning](https://lup.lub.lu.se/student-papers/search/publication/9022444). A few examples that illustrate the usage of this package are also included.

# Installation

Installation will be done on Windows 10 using [Anaconda](https://www.anaconda.com), more specifically using the Anaconda Prompt. The code and the used packages require Python 3.5.6.

## Part 1: PyFMI

1. Open Anaconda Prompt.

2. Create a new environment (in this case called *HaldexROM*)
    ```
    conda create -n HaldexROM python==3.5.6
    ```

3. Activate environment by typing `conda activate HaldexROM`.

4. Install packages
    ```
    conda install -c anaconda cython
    conda install -c anaconda numpy
    conda install -c anaconda scipy
    conda install -c anaconda lxml
    conda install -c anaconda sympy
    conda install -c conda-forge assimulo
    conda install -c conda-forge pyfmi
    ```
    The versions we used for each package is shown in this table. Note that other versions might work as well.

    | package  | version        |
    |----------|----------------|
    | cython   | 0.28.5/0.29.16 |
    | numpy    | 1.15.2/1.18.3  |
    | scipy    | 1.1.0/1.4.1    |
    | lxml     | 4.2.5          |
    | assimulo | 2.9            |
    | pyfmi    | 2.4.0          |

5. Update package for plots
    ```
    conda update matplotlib
    ```

## Part 2: TensorFlow

1. Update packages
    ```
    python -m pip install --upgrade pip
    pip install attrs --upgrade
    pip install jsonschema --upgrade
    pip install setuptools --upgrade
    ```

2. Install TensorFlow. Version `2.3.0rc0` or later is required. For more details see the official [installation guide]([https://www.tensorflow.org/install/](https://www.tensorflow.org/install/) and the [PyPI page](https://pypi.org/project/tensorflow/#history) where you can grab a specific version of the package.
