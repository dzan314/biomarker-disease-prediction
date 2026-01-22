import pandas as pd
import xgboost as xgb
import seaborn as sns
import lightgbm as lgbm
import matplotlib.pyplot as plt
import shap as shap
import feature_en as fen
# ==========================================
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

demo = pd.read_csv("demographic.csv")
labs = pd.read_csv("labs.csv")
diet = pd.read_csv("diet.csv")
questions = pd.read_csv("questionnaire.csv")
examination = pd.read_csv("examination.csv")

# merge datasets into one
df = demo \
    .merge(labs, on="SEQN", how="left") \
    .merge(diet, on="SEQN", how="left") \
    .merge(questions, on="SEQN", how="left") \
    .merge(examination, on="SEQN", how="left")

# cleaning the data

df = df.replace([7, 8, 9], pd.NA).infer_objects(copy=False) # droping (replacing with "NaN") invalid survey-type values 

df = df[df["DIQ010"].isin([1, 2])] # drop any value in column DIQ010 (was diabetes diagnosed) that's not 1 or 2 (yes/no)
df["diabetes"] = (
    df["DIQ010"]
    .replace({1:1, 2:0})
    .astype("int64")
    ) # convert to binary - diabetes positive = 1, negative = 0

df = df.dropna(subset=["diabetes"]) # drop missing values in target column

df = df[df["RIDAGEYR"] >= 20] # only take aged 20 or more

df["SBP_mean"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1) # Systolic Blood Pressure - mean out of 3 samples (in NHANES examinations its examined 3 times per participant)
df["DBP_mean"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)# Diatolic Blood Pressure - mean out of 3 samples (in NHANES examinations its examined 3 times per participant)

# eliminating any potential tail values - borderline insane readings (normalizing the scale of measurementrs)
df = df[
    (df["BMXBMI"].between(10, 80)) &
    (df["SBP_mean"].between(70, 250)) &
    (df["DBP_mean"].between(40, 150))
]

def selection_run(n_experiments: int, p_features: int):

    """
    Function creates random subsets from the feature pool and runs the XGBoost model on them. \n
    Returns the subset with the biggest ROC-AUC value, along with its PR-AUC value.
    
    :param n_experiments: number of random feature subsets from the feature pool of size = 30
    :param p_features: size (number of features) of these random subsets
    """

    feat_roc_dict = {}
    all_subsets = []
    roc_scores = []
    pc_scores = []

    FEATURE_POOL = fen.feature_pool_init(
        fen.ALL_FEATURES,
        df.columns
    )

    for i in range(n_experiments):

        risk_features = fen.random_subset(FEATURE_POOL, size=p_features) # creating random subsets from the feature pool
        all_subsets.append(risk_features)

        X_risk = df[risk_features]
        y = df["diabetes"] # set target

        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X_risk, y, test_size=0.2, stratify=y, random_state=42
        )

        label_encoders = {}

        for col in fen.CATEGORICAL_FEATURES:
            if col in Xr_train.columns:
                label = LabelEncoder()
                Xr_train[col] = label.fit_transform(Xr_train[col].astype(str))
                Xr_test[col]  = label.transform(Xr_test[col].astype(str))
                label_encoders[col] = label


        Xr_train = Xr_train.apply(pd.to_numeric, errors="coerce")
        Xr_test  = Xr_test.apply(pd.to_numeric, errors="coerce")

        median_values = Xr_train.median()

        # empty cells filled with median values
        Xr_train = Xr_train.fillna(median_values)
        Xr_test  = Xr_test.fillna(median_values)

        # the XGBoost model
        xgb_risk = xgb.XGBClassifier(
            n_estimators=303,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        xgb_risk.fit(Xr_train, yr_train) # fitting on training data

        yr_pred_proba = xgb_risk.predict_proba(Xr_test)[:, 1] # predict_proba returns a 2D array, where the second column is the positive score for diabetes (hence the slice)

        roc = roc_auc_score(yr_test, yr_pred_proba)
        pc = average_precision_score(yr_test, yr_pred_proba)
        roc_scores.append(roc)
        pc_scores.append(pc)

        feat_roc_dict[roc_scores[i]] = all_subsets[i] # update the dictionary with roc - feature subset pairs
        largest_roc = max(feat_roc_dict.keys()) # get the largest roc value
        best_set = feat_roc_dict[largest_roc] # get the feature subset that's responsible for this roc value

    return (best_set, largest_roc, pc_scores[i]) 

if __name__ == "__main__":

    print(f"BEST ROC-AUC: {selection_run(50, 15)}")
