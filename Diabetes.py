
Diabetes.py
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import missingno as msno
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pip
!pip install catboost
from catboost import CatBoostRegressor
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.pandas.set_option('display.max_rows', None)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("C:/Users/Gavita/Desktop/dsmlbcc/Diabets/DataSets/diabetes.csv")


def general(dataframe):
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.info())
    print(dataframe.columns)
    print(dataframe.isnull().values.any())

general(df)

df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
sns.countplot(x='Outcome', data=df)
plt.show()

df["Outcome"].describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T


df.groupby("Outcome").size()
df["Outcome"].value_counts()*100/len(df)

df.hist(figsize=(20,20))
plt.show()

df.corr()

#Missing Value Analysis

df.isnull().sum()
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
# These values cannot be 0 in living human body. I replaced 0 with Nan

msno.bar(df)
plt.show()

df["Glucose"].hist(edgecolor="black")
plt.show()
df["Glucose"].describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T

# We should calculate two medians for each variable, one for sicks and one for not sicks
def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

median_target("Glucose")
median_target("BloodPressure")
median_target("SkinThickness")
median_target("Insulin")
median_target("BMI")

columns = df.columns
columns = columns.drop("Outcome")

for col in columns:
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]

df.isnull().sum()

#Outliers

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        print(variable, "yes")


for col in df.columns:
    has_outliers(df,col)

outlier_thresholds(df,"SkinThickness")
df["SkinThickness"].describe([0.10,0.25,0.5,0.75,0.90])
df["SkinThickness"].min()

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df.columns:
    replace_with_thresholds(df, col)

for col in df.columns:
    has_outliers(df,col)

# Feature Engineering
df["Glucose"].describe()
df["Glucose"]=df["Glucose"].astype("int")
df["Glucose"].sort_values(ascending=False)
NewGlucose = pd.Series(["Hypoglycemia", "Normal", "PreDiabetes", "Diabetes"], dtype = "category")
df.loc[df["Glucose"]<=70, "NewGlucose"]="Hypoglycemia"
df.loc[(df["Glucose"]>70)&(df["Glucose"]<=100), "NewGlucose"]="Normal"
df.loc[(df["Glucose"]>100)&(df["Glucose"]<=125), "NewGlucose"]="PreDiabetes"
df.loc[(df["Glucose"]>125),"NewGlucose"]="Diabetes"
df.head()

df["BMI"].describe()
NewBMI = pd.Series(["LowWeight", "Normal","Overweight","Obesity"], dtype = "category")
df["NewBMI"]=NewBMI
df.loc[df["BMI"]<= 18.5,"NewBMI"]=NewBMI[0]
df.loc[(df["BMI"]>18.5)&(df["BMI"]<=24.99),"NewBMI"]=NewBMI[1]
df.loc[(df["BMI"]>24.99)&(df["BMI"]<=29.99),"NewBMI"]=NewBMI[2]
df.loc[(df["BMI"]>29.99),"NewBMI"]=NewBMI[3]
df.head()


df["Insulin"].describe()
NewInsulin=pd.Series(["Normal","Diabetes"],dtype="category")
df["NewInsulin"]=NewInsulin
df.loc[df["Insulin"]<=126,"NewInsulin"]=NewInsulin[0]
df.loc[df["Insulin"]>126,"NewInsulin"]=NewInsulin[1]
df.head()

df["BloodPressure"].describe()
NewBloodPress=pd.Series(["Normal","Diabetes"],dtype="category")
df["NewBloodPress"]=NewBloodPress
df.loc[df["BloodPressure"]<80,"NewBloodPress"]=NewBloodPress[0]
df.loc[df["BloodPressure"]>=80,"NewBloodPress"]=NewBloodPress[1]
df.head()

df["Pregnancies"].describe([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
df["Pregnancies"].value_counts()
df.groupby("Pregnancies").agg({"Age":"mean"})
df.loc[df["Pregnancies"]>df["Pregnancies"].quantile(0.90), "Pregnancies"]= df["Pregnancies"].quantile(0.90)
df.loc[df["Pregnancies"]<=3, "Pregnancies"]=3
df.loc[(df["Pregnancies"]>3)&(df["Pregnancies"]<=6), "Pregnancies"]=2
df.loc[(df["Pregnancies"]>6),"Pregnancies"]=1
df["Pregnancies"].describe()

df.head()

df["SkinThickness"].describe([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
df.groupby("SkinThickness").agg({"Age":"mean"})
df["Age"].min()
SkinAgeIndx=pd.Series(["VeryLow", "Low","Optimal","ModHigh","High"], dtype = "category")
df["SkinAgeIndx"]=SkinAgeIndx
df.loc[(df["SkinThickness"]<16)&(df["Age"]<=29),"SkinAgeIndx"]=SkinAgeIndx[0]
df.loc[(df["SkinThickness"]<17)&(df["Age"]>29)|(df["Age"]<=39),"SkinAgeIndx"]=SkinAgeIndx[0]
df.loc[(df["SkinThickness"]<18)&(df["Age"]>39)|(df["Age"]<=49),"SkinAgeIndx"]=SkinAgeIndx[0]
df.loc[(df["SkinThickness"]<19)&(df["Age"]>49)|(df["Age"]<=59),"SkinAgeIndx"]=SkinAgeIndx[0]
df.loc[(df["SkinThickness"]<20)&(df["Age"]>59),"SkinAgeIndx"]=SkinAgeIndx[0]

df.loc[(df["SkinThickness"]>=16)|(df["SkinThickness"]<20)&(df["Age"]<=29),"SkinAgeIndx"]=SkinAgeIndx[1]
df.loc[(df["SkinThickness"]>=17)|(df["SkinThickness"]<21)&(df["Age"]>29)|(df["Age"]<=39),"SkinAgeIndx"]=SkinAgeIndx[1]
df.loc[(df["SkinThickness"]>=18)|(df["SkinThickness"]<22)&(df["Age"]>39)|(df["Age"]<=49),"SkinAgeIndx"]=SkinAgeIndx[1]
df.loc[(df["SkinThickness"]>=19)|(df["SkinThickness"]<23)&(df["Age"]>49)|(df["Age"]<=59),"SkinAgeIndx"]=SkinAgeIndx[1]
df.loc[(df["SkinThickness"]>=20)|(df["SkinThickness"]<24)&(df["Age"]>59),"SkinAgeIndx"]=SkinAgeIndx[1]

df.loc[(df["SkinThickness"]>=20)|(df["SkinThickness"]<29)&(df["Age"]<=29),"SkinAgeIndx"]=SkinAgeIndx[2]
df.loc[(df["SkinThickness"]>=21)|(df["SkinThickness"]<30)&(df["Age"]>29)|(df["Age"]<=39),"SkinAgeIndx"]=SkinAgeIndx[2]
df.loc[(df["SkinThickness"]>=22)|(df["SkinThickness"]<31)&(df["Age"]>39)|(df["Age"]<=49),"SkinAgeIndx"]=SkinAgeIndx[2]
df.loc[(df["SkinThickness"]>=23)|(df["SkinThickness"]<32)&(df["Age"]>49)|(df["Age"]<=59),"SkinAgeIndx"]=SkinAgeIndx[2]
df.loc[(df["SkinThickness"]>=24)|(df["SkinThickness"]<33)&(df["Age"]>59),"SkinAgeIndx"]=SkinAgeIndx[2]

df.loc[(df["SkinThickness"]>=29)|(df["SkinThickness"]<31)&(df["Age"]<=29),"SkinAgeIndx"]=SkinAgeIndx[3]
df.loc[(df["SkinThickness"]>=30)|(df["SkinThickness"]<32)&(df["Age"]>29)|(df["Age"]<=39),"SkinAgeIndx"]=SkinAgeIndx[3]
df.loc[(df["SkinThickness"]>=31)|(df["SkinThickness"]<33)&(df["Age"]>39)|(df["Age"]<=49),"SkinAgeIndx"]=SkinAgeIndx[3]
df.loc[(df["SkinThickness"]>=32)|(df["SkinThickness"]<34)&(df["Age"]>49)|(df["Age"]<=59),"SkinAgeIndx"]=SkinAgeIndx[3]
df.loc[(df["SkinThickness"]>=33)|(df["SkinThickness"]<35)&(df["Age"]>59),"SkinAgeIndx"]=SkinAgeIndx[3]

df.loc[(df["SkinThickness"]>=31)&(df["Age"]<=29),"SkinAgeIndx"]=SkinAgeIndx[4]
df.loc[(df["SkinThickness"]>=32)&(df["Age"]>29)|(df["Age"]<=39),"SkinAgeIndx"]=SkinAgeIndx[4]
df.loc[(df["SkinThickness"]>=33)&(df["Age"]>39)|(df["Age"]<=49),"SkinAgeIndx"]=SkinAgeIndx[4]
df.loc[(df["SkinThickness"]>=34)&(df["Age"]>49)|(df["Age"]<=59),"SkinAgeIndx"]=SkinAgeIndx[4]
df.loc[(df["SkinThickness"]>=35)&(df["Age"]>59),"SkinAgeIndx"]=SkinAgeIndx[4]
df["SkinAgeIndx"].describe()
df.head()

df["DiabetesPedigreeFunction"].describe()
df.groupby("Outcome").agg({"DiabetesPedigreeFunction":"median"})
df["DiabetesPedigreeFunction"].max()

df=pd.get_dummies(df,columns =["NewBMI","NewInsulin", "NewGlucose","SkinAgeIndx","NewBloodPress"], drop_first = True)
df.head()


#Modelling

#Logistik Reg. Modelling

y = df["Outcome"]
X = df.drop(["Outcome",'NewBMI_Normal','NewBMI_Obesity','NewBMI_Overweight','NewInsulin_Normal',
             'NewGlucose_Hypoglycemia','NewGlucose_Normal','NewGlucose_PreDiabetes','SkinAgeIndx_Low',
             'SkinAgeIndx_ModHigh','SkinAgeIndx_Optimal','SkinAgeIndx_VeryLow','NewBloodPress_Normal'], axis = 1)

X.head()
y.head()

log_model=LogisticRegression().fit(X,y)
log_model.intercept_
log_model.coef_

y_pred=log_model.predict(X)
accuracy_score(y,y_pred)

cross_val_score(log_model, X, y, cv=10).mean()
print(classification_report(y, y_pred))

log_model = LogisticRegression().fit(X,y)
y_pred = log_model.predict(X)
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))


logit_roc_auc = roc_auc_score(y, log_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, log_model.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# RandomForest

rf_model=RandomForestClassifier(random_state=12345).fit(X, y)

cross_val_score(rf_model,X,y,cv=10).mean()

rf_params = {"n_estimators": [100,200,500,1000],
             "max_features": [3,5,7],
             "min_samples_split": [2,5,10,30],
             "max_depth": [3,5,8, None]}

rf_model=RandomForestClassifier(random_state=12345)

gs_cv = GridSearchCV(rf_model,
                     rf_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

gs_cv.best_params_

rf_tuned = RandomForestClassifier(**gs_cv.best_params_)
cross_val_score(rf_tuned, X, y, cv=10).mean()


# LightGBM

lgbm = LGBMClassifier(random_state=12345)
cross_val_score(lgbm, X, y, cv=10).mean()

# model tuning
lgbm_params={"learning_rate":[0.01,0.03,0.05,0.1,0.5],
               "n_estimators":[500,1000,1500],
               "max_depth":[3,5,8]}

LGBMClassifier()

gs_cv=GridSearchCV(lgbm,
                   lgbm_params,
                   cv=5,
                   n_jobs=-1,
                   verbose=2).fit(X, y)

gs_cv.best_params_

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
cross_val_score(lgbm_tuned, X, y, cv=10).mean()

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Variable Significance Scores')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()


models = [('LR', LogisticRegression()),
          ('RF', RandomForestClassifier()),
          ("LightGBM", LGBMClassifier())]


results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)