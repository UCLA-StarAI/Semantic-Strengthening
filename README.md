# Semantic Strengthening of Neuro-Symbolic Losses

This repository holds the code for the AISTATS 2023 paper [Semantic Strengthening of Neuro-Symbolic Losses]
by Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck

We introduce Semantic Strengthening,an approach for scalable neuro-symbolic learning. We approach the problem by first assuming the constraint decomposes conditioned on the features learned by the network. We iteratively strengthen our approximation, restoring the dependence between the constraints most responsible for degrading the quality of the approximation.

-------------------- 

## Installation
```
conda env create -f environment.yml
```

and if you encounter a pypsdd related error, running the following should solve the issue
```
pip install -vvv --upgrade --force-reinstall --no-binary :all: --no-deps pysdd
```

## Commands
Each of the three tasks includes a .sh script.
