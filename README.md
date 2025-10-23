# Performative Risk Control Experiments

## Setup

To run the simulations, first set up a conda environment.
```
conda create -n rcpp python=3.12
conda activate rcpp
pip install -r requirements.txt
```

Download/unzip the [Kaggle credit scoring dataset](https://www.kaggle.com/c/GiveMeSomeCredit/data) to `applications/credit_scoring/data/`.

## How to run

To run the credit scoring experiments, run:
```
python -m applications.credit_scoring.expected_loss
python -m applications.credit_scoring.quantile
```

To run the forecasting experiment, see the Jupyter notebook located at `applications/forecasting/expected_loss.ipynb`.

Running the above should create a new folder called `figures/` within their respective folders


## References

[Credit Fusion and Will Cukierski. Give Me Some Credit. https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.](https://www.kaggle.com/c/GiveMeSomeCredit/data)
