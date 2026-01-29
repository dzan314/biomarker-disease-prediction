# Blood biomarker disease prediction

## Overview

Type II diabetes predictor

Risk prediction

## Features

TODO --> pipelines , launching

## Data Set

NHANES (National Health and Nutrition Examination Survey) dataset from 2013-2014 (Kaggle)

## Methods

Main model: XGBoost

Run Boruta on the features pool, using Random Forest model

This resulted in obtaining 17 relevant features, however they get outperformed by the randomly sampled features (p=15, see below)

Thus, SHAP is a next step

--------------------------------

After feature selection process:

For features set X = ['LBDLDL', 'PAQ605', 'RIDAGEYR', 'DR1TFIBE', 'SLD010H', 'DMDEDUC2', 'INDHHIN2', 'DR1TSUGR', 'RIDRETH1', 'BMXHT', 'LBXSCR', 'BMXWAIST', 'ALQ101', 'LBXSTR', 'SBP_mean']

ROC AUC: 0.8423920711060948 --> 0.8572058408577878  (+ 1.7% )

PR-AUC: 0.4424574511709692 --> 0.47388243098907745  (+ 7% )

Regarding the randomised feature selection - the pool of all features is set to size = 30. TODO --> new experimental file, with bigger pool size and different subset sizes

Second (comparision) model: LightGBM <-- TODO

## Results

TODO:

Check Shepley's values + Sensitivity + Calibration (curve) plot

## Usage

Install dependencies if needed:

``` bash
pip install pandas
pip install scikit-learn
pip install xgboost
pip install lightgbm
pip install seaborn
pip install shap
```

## Future Work

## License

MIT License
