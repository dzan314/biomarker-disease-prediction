import random

ALL_FEATURES = [
"RIDAGEYR", # Age (years)
"RIAGENDR", # Sex
"RIDRETH1", # Race / ethnicity
"DMDEDUC2", # Education level
"INDHHIN2", # Household income
"DMDMARTL", # Marital status
"BMXBMI", # Body Mass Index
"BMXWAIST", # Waist circumference
"BMXWT", # Body weight
"SBP_mean", # Mean systolic blood pressure
"DBP_mean", # Mean diastolic blood pressure
"BPQ020", # Ever told you had hypertension
"BMXHT", # Height
"LBXSCH", # Total cholesterol
"LBDHDL", # HDL cholesterol
"LBDLDL", # LDL cholesterol
"LBXSTR", # Triglycerides
"LBXSUA", # Uric acid
"LBXSCR", # Creatinine (kidney function)
"PAQ605", # Moderate physical activity
"PAQ650", # Vigorous physical activity
"SMQ020", # Smoking status
"ALQ101", # Ever drank alcohol
"SLD010H", # Sleep duration
"SLQ050", # Trouble sleeping
"DR1TKCAL", # Total daily calories
"DR1TSUGR", # Total sugar intake
"DR1TFIBE", # Fiber intake
"DPQ020", # Depressive symptoms
"MCQ010" #  General health condition
]

CATEGORICAL_FEATURES = [
    "RIAGENDR", # Age
    "RIDRETH1", # Ethnicy
    "DMDEDUC2", # Education level
    "INDHHIN2" # Household income
]

# ADD DOCSTRINGS

def feature_pool_init(all_features, available_col):
    return [f for f in all_features if f in available_col]

def random_subset(pool, size=15):
    if size > len(pool):
        raise ValueError("Subset size cannot be larger than feature pool!")
    return random.sample(pool, size)






