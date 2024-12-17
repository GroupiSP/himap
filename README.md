# hsmm

Package for HMM and HSMM models including prognostics

## Table of Contents

- [Configuration and Installation](#installation)
- [Data Structure](#structure)
- [Example](#example)
- [Contributors](#contributors)

## Installation

The steps to configure and install the packages are the following:

1. Create an Anaconda environment and activate it.

  Step 1a

```
conda create -n hmm_env python=3.9.16
```

  Step 1b

```
conda activate hmm_env
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


  Step 2c
```
cd hsmm_dev
```

3. The dependencies can be installed using the requirements.txt file
```
pip install -r requirements.txt
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

          ├── example_data/                                             -- Required      
              └── test_FD001_disc_20_mod.csv                            -- Required
              └── train_FD001_disc_20_mod.csv                           -- Required

          ├── results/                                                  -- Automatically generated      
              ├── dictionaries                                          -- Automatically generated
              ├── figures/                                              -- Automatically generated
              ├── models/                                               -- Automatically generated
```

## Example

To describe how to train and use the HMM and HSMM models, we show an example below. To run the code from the Anaconda terminal with default values, go to the `hmm` folder inside the `hsmm_dev` directory and run the `main.py` file via the commands:

```
cd hmm
```

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
