import pandas as pd
import sklearn as sklearn
import feature_en as fen
import numpy as np
# ===================================
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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

risk_features = fen.feature_pool_init(
        fen.ALL_FEATURES,
        df.columns
    )

X = df[risk_features]
y = df["diabetes"] # set target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

label_encoders = {}

for col in fen.CATEGORICAL_FEATURES:
    label = LabelEncoder()
    X_train[col] = label.fit_transform(X_train[col].astype(str))
    X_test[col] = label.fit_transform(X_test[col].astype(str))            <---- # There is a data leakage here, need to FIX THIS (fit independent of transform)
    label_encoders[col] = label

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test  = X_test.apply(pd.to_numeric, errors="coerce")

median_values = X_train.median()

# empty cells filled with median values
X_train = X_train.fillna(median_values)
X_test  = X_test.fillna(median_values)

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(np.array(X_train), np.array(y_train))

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(np.array(X_train))

if __name__ == "__main__":
    confirmed_features = X_train.columns[feat_selector.support_]
    print(list(confirmed_features))
