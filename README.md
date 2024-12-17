# hsmm
Package for HMM and HSMM models including prognostics

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

## Example
>**Note**
>This is the beta version of the example, where the test_cmapss.py file is used. We need to add the parser.args method to pull inputs from the cmd and have flags for mc example or cmapss example and for hmm or hsmm.

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
