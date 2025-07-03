# Implementation for Performative Risk Control

This repo implements simulations to analyze and demonstrate performative risk control in credit scoring models.

```
uv sync
```


## Setup

To run the credit scoring simulations,
```
conda create -n rcpp python=3.12
conda activate rcpp
pip install -r requirements.txt
```

Download/unzip the [Kaggle credit scoring dataset](https://www.kaggle.com/c/GiveMeSomeCredit/data) to `applications/credit_scoring/data/` and [Bitcoin historical data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) to `applications/market_making/data/`. Then, run `python -m applications.market_making.subsample_data` to generate "./applications/market_making/data/btcusd_1-min_data_filtered.csv".




## How to run

Finally, to replicate the expected loss and quantile experiments, run the following:
```
python -m applications.credit_scoring.expected_loss
python -m applications.credit_scoring.quantile
```

Running these commands should create a new folder called `figures/` inside the `credit_scoring` directory.


## References

[Credit Fusion and Will Cukierski. Give Me Some Credit. https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.](https://www.kaggle.com/c/GiveMeSomeCredit/data)

[Zielak. Bitcoin Historical Data. https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data, 2025. Kaggle.](https://www.kaggle.com/c/GiveMeSomeCredit/data). Accessed 07/03/2025.
