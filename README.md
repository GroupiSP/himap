# HiMAP:  Hidden Markov for Advanced Prognostics

Package for HMM and HSMM models including prognostics

## Table of Contents

- [Configuration and Installation](#installation)
- [Data Structure](#structure)
- [Example](#example)
- [Contributors](#contributors)

## Requirements
> [!WARNING]
> A C++ compiler is required to build the .pyx files.

### Windows users:
Microsoft Visual C/C++ (MSVC) 14.0 or higher is required to build the .pyx files.

https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

(Download Build Tool - After Visual Studio Installer is ready, choose Desktop development with C++)

### Linux users:
The GNU C Compiler (gcc) is usually present. Next to a C compiler, Cython requires the Python header files. 
On Ubuntu or Debian run the following command:

```
sudo apt-get install build-essential python3-dev
```

>[!Note]
>For more information refer to the Cython package documentation:
>
>https://cython.readthedocs.io/en/latest/src/quickstart/install.html

## Installation
The steps to configure and install the packages are the following:

1. Create an Anaconda environment and activate it.

  Step 1a

```
conda create -n himap python=3.9.16
```

  Step 1b

```
conda activate himap
```


2. This repository can be directly pulled through GitHub by the following commands:

  Step 2a
```
conda install git
```

  Step 2b
```
git clone https://github.com/thanoskont/hsmm_dev.git
```
>**Note**
>This command will be the correct when everything is merged to the main branch. For our use now, in order to pull directly from the hmm_add branch use:
>``` git clone -b hmm_add --single-branch https://github.com/thanoskont/hsmm_dev.git ```


  Step 2c
```
cd hsmm_dev
```

3. The dependencies can be installed using the requirements.txt file
```
pip install -r requirements.txt
```

4. To compile the Cython code, run the following commands:
   
  Step 4a
```
cd hmm/cython_build
```

  Steb 4b
```
python setup.py build_ext --inplace
```

```
cd ..
```


## Structure


```
../hsmm_dev/
      └── LICENSE
      └── README.md
      └── requirements.txt
    
      ├── hmm/                                                          -- Required
          └── ab.py                                                     -- Required
          └── base.py                                                   -- Required
          └── main.py                                                   -- Required
          └── plot.py                                                   -- Required
          └── smoothed.pyd                                              -- Required
          └── utils.py                                                  -- Required
          
          ├── cython_build/                                             -- Required      
              └── fwd_bwd.pyx                                           -- Required
              └── setup.py                                              -- Required


          ├── example_data/                                             -- Required      
              └── test_FD001_disc_20_mod.csv                            -- Required
              └── train_FD001_disc_20_mod.csv                           -- Required

          ├── results/                                                  -- Automatically generated      
              ├── dictionaries                                          -- Automatically generated
              ├── figures/                                              -- Automatically generated
              ├── models/                                               -- Automatically generated
```

## Example
>**Note**
>This is the beta version of the example, where the test_cmapss.py file is used. We need to add the parser.args method to pull inputs from the cmd and have flags for mc example or cmapss example and for hmm or hsmm.

To describe how to train and use the HMM and HSMM models, we show an example below. To run the code from the Anaconda terminal with default values, go to the `hmm` folder inside the `hsmm_dev` directory and run the `main.py` file via the commands:


```
python main.py
```

This runs the HMM model for the C-MAPSS dataset by default and fits the best model utilizing the Bayesian Information Criterion.

If you want to fit the HSMM model to the C-MAPSS data run the command:

```
python main.py --hsmm True 
```

If you want to run the example utilizing Monte Carlo Sampling generated data run the command:

```
python main.py --mc_sampling True
```

See the `main.py` file for different existing variables and options.

### Results

The results are saved inside the directory `../hsmm_dev/hmm/results/`

## Contributors

- [Thanos Kontogiannis](https://github.com/thanoskont)
- [Mariana Salinas-Camus](https://github.com/mariana-sc)

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
