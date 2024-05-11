# Python Foundations Repository

Welcome to the Python Foundations Repository. This project aims to build a comprehensive understanding of Python and its essential libraries, particularly for machine learning and artificial intelligence. The materials are designed for students and professionals who want to strengthen their grasp on widely-used libraries, ensuring a solid Python foundation.

## Repository Structure

### Absolute Basics
- Contains resources sourced from ["Python Crash Course"](https://ehmatthes.github.io/pcc/), providing an in-depth understanding of core Python concepts. You will find:
  - Scripts ranging from simple introductory examples to practical exercises with data types, control structures, functions, and object-oriented programming.

### Numpy
- Originally developed, this section introduces numerical computing using NumPy:
  - Jupyter notebooks and comprehensive guides to manipulating NumPy arrays.
  - Supplemental materials like cheat sheets and setup instructions for effective use.

### Simple_NN
- A straightforward neural network implementation for learning and practice:
  - Organized subdirectories for datasets, source code, and trained models, offering practical training, validation tools, and modularized network components.

## Future Enhancements

- **Additional Libraries**: Plans to expand with practical exercises on other critical Python libraries like Pandas (for data manipulation), Matplotlib/Seaborn (for data visualization), and Scikit-learn (for machine learning).
- **Deep Learning Frameworks**: Future sections will include deep learning frameworks such as TensorFlow and PyTorch, providing a hands-on approach to building advanced neural networks.

## Getting Started

### Environment Setup

1. **Remove any existing environment (if needed):**
   ```
   conda remove --name learnpy --all
   ```

2. **Create and activate a new environment:**
   ```
   conda create -n learnpy python=3.11 -y
   conda activate learnpy
   ```

3. **Install required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Install [Jupyter Notebook](https://jupyter.org/install) and Extensions:**
   - Installing Jupyter Notebook (default):
     ```
     conda install -c conda-forge notebook
     conda install -c conda-forge nb_conda_kernels
     ```
   - Installing Jupyter Lab (optional):
     ```
     conda install -c conda-forge jupyterlab
     conda install -c conda-forge nb_conda_kernels
     ```
   - Installing Jupyter Notebook extensions (optional but recommended):
     ```
     conda install -c conda-forge jupyter_contrib_nbextensions
     ```

### Usage

Once the environment is set up, explore the [**Absolute Basics**](#absolute-basics) section to establish a fundamental understanding of Python syntax. Then proceed to the [**Numpy**](#numpy) and [**Simple_NN**](#simple_nn) directories to further delve into numerical computing and machine learning.

### Note
This repository leverages the `i2dl` environment from TUM for compatibility and reproducibility. Refer to specific instructions if you encounter any issues.

## License and Acknowledgments

- The "Absolute Basics" content is sourced from "Python Crash Course."
- NumPy and other practical exercises are originally created.
- [MIT license](https://opensource.org/license/mit).

## Conda Installation Guide

If you're new to Conda, it's a powerful package manager and environment manager that you can use with command line commands on macOS, Linux, and Windows. Conda makes it easy to install, run, and upgrade complex software dependencies. Here is how you can install Conda:

- Visit the [Anaconda website](https://www.anaconda.com/products/individual) and download the Anaconda Installer for your operating system.
- Follow the on-screen instructions to complete the installation. For a more detailed guide, you can check the official [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

**First Published:** May 2024
