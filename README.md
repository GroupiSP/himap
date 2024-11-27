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

To describe how to train and use the HMM and HSMM models, we show an example below. To run the code from the Anaconda terminal with default values, go to the `hmm` folder inside the `hsmm_dev` directory and run the `test_cmapss.py` file via the commands:

```
cd hmm
```

```
python test_cmapss.py
```

This runs the HMM model for the C-MAPSS dataset by default.

### Results

The results are saved inside the directory `../hsmm_dev/hmm/results/`
