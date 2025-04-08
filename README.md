# Learning Augmented Robust Algorithmic Recourse

This is a joint work of [Kshitij Kayastha](https://github.com/kshitij-kayastha), [Shahin Jabbari](https://shahin-jabbari.github.io/), and [Vasilis Gkatzelis](https://www.cs.drexel.edu/~vg399/).

## Installation instructions

1. Clone the repo.

2. Create and activate a virtual environment

```shell
cd "learning-augmented-robust-recourse"
python -m venv .env
source .env/bin/activate
```

3. Install the dependencies

```shell
pip install requirements.txt
```

## Reproducing Results

You can replicate the experiments described in the paper by running the notebooks in the `experiments/` directory. Notebooks that end with 'fig' visualize the results saved in the `results/` directory. You can overwrite the results by running the corresponding notebook without the 'fig'. For example, you can run the code in `rob_cont_tradeoff_lr.ipynb` with your choice of parameters and overwrite the results, then visualize the results using `rob_cont_tradeoff_fig.ipynb` notebook. 

## Datasets
- <b>German Credit Dataset</b>
    - Link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    - Features used: `duration`, `amount`, `age`, `personal_status_sex`.

- Small Business Dataset
    - Link: https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set
    - Features used: `Zip`, `NAICS`, `ApprovalDate`, `ApprovalFY`, `Term`, `NoEmp`, `NewExist`, `CreateJob`, `RetainedJob`, `FranchiseCode`, `UrbanRural`, `RevLineCr`, `ChgOffDate`, `DisbursementDate`, `DisbursementGross`, `ChgOffPrinGr`, `GrAppv`, `SBA_Appv`, `New`, `RealEstate`, `Portion`, `Recession`, `daysterm`, `xx`.
