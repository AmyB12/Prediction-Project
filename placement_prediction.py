# IMPORTING LIBRARIES
import matplotlib.pyplot
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import StrMethodFormatter
# import matplotlib.ticker as mtick
import numpy as np
# import seaborn as sns
# import pandas_profiling
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import statsmodels.api as sm
# from sklearn.metrics import mean_absolute_error, r2_score
# import eli5
from eli5.sklearn import PermutationImportance

# print (matplotlib.pyplot.get_backend())

# LOADING DATASET
data = pd.read_csv("Placement_Data.csv")
data.drop("sl_no", axis=1, inplace=True)  # remove the indexing from data
pd.set_option("display.max_rows", None, "display.max_columns", None)  # print all the column when called throughout code
# print(data.head()) # first 5 rows
# print(data.shape) # show how many rows and columns
# print(data.describe()) # show the stats info for data
# print(data.isnull().sum()) # shows all the columns and if there are any null/ missing data

# PANDAS PROFILER'S INTERACTIVE REPORT
# data2 = pandas_profiling.ProfileReport(data)
# data2.to_file('pandas_profile.html')

# EXPLORING DATA BY EACH FEATURES
### Feature: Gender ###
# print(data.gender.value_counts())
matplotlib.use('TkAgg')
# sns.pairplot(data.drop("salary", axis=1), hue="status") # compares all except salary
# sns.countplot("gender", hue="status", data=data) # counts how many get placed or not between genders
# sns.kdeplot(data.salary[data.gender=="M"]) # kernel distribution plot (probability density function)
# sns.kdeplot(data.salary[data.gender=="F"])
# plt.legend(["Male", "Female"])
# plt.xlabel("Salary(100k)")
# plt.figure(figsize=(18, 6)) # box plot
# sns.boxplot("salary", "gender", data=data)
### DOES SECONDARY EDUCATION AFFECT PLACEMENT ###
# sns.kdeplot(data.ssc_p[data.status=="Placed"]) # kernel distribution plot
# sns.kdeplot(data.ssc_p[data.status=="Not Placed"])
# plt.legend(["Placed", "Not Placed"])
# plt.xlabel("Secondary Education Percentage")
# sns.countplot("ssc_b", hue="status", data=data)
# plt.figure(figsize=(18, 6))
# sns.boxplot("salary", "ssc_b", data=data)
# sns.lineplot("ssc_p", "salary", hue="ssc_b", data=data)
### DOES HIGHER EDUCATION AFFECT PLACEMENT ###
# sns.kdeplot(data.hsc_p[data.status=="Placed"])
# sns.kdeplot(data.hsc_p[data.status=="Not Placed"])
# plt.legend(["Placed", "Not Placed"])
# plt.xlabel("Higher Secondary Education Percentage")
# sns.countplot("hsc_b", hue="status", data=data)
# sns.countplot("hsc_s", hue="status", data=data)
# plt.figure(figsize=(18, 6))
# sns.boxplot("salary", "hsc_b", data=data)
# sns.lineplot("hsc_p", "salary", hue="hsc_b", data=data)
# plt.figure(figsize=(18, 6))
# sns.boxplot("salary", "hsc_s", data=data)
# sns.lineplot("hsc_p", "salary", hue="hsc_s", data=data)
### DOES UNDER GRAD AFFECT PLACEMENTS ###
# sns.kdeplot(data.degree_p[data.status=="Placed"])
# sns.kdeplot(data.degree_p[data.status=="Not Placed"])
# plt.legend(["Placed", "Not Placed"])
# plt.xlabel("Under Grad Education Percentage")
# sns.countplot("degree_t", hue="status", data=data)
# plt.figure(figsize=(18, 6))
# sns.boxplot("salary", "degree_t", data=data)
# sns.lineplot("degree_p", "salary", hue="degree_t", data=data)
### DOES WORK EXPERIENCE AFFECT PLACEMENTS ###
# sns.countplot("workex", hue="status", data=data)
# plt.figure(figsize=(18, 6))
# sns.boxplot("salary", "workex", data=data)
### Employability percentage ###
# sns.kdeplot(data.etest_p[data.status=="Placed"])
# sns.kdeplot(data.etest_p[data.status=="Not Placed"])
# plt.legend(["Placed", "Not Placed"])
# plt.xlabel("Employability test Percentage")
# sns.lineplot("etest_p", "salary", data=data)
### POST GRAD SPECIALISATION ###
# sns.countplot("specialisation", hue="status", data=data)
# plt.figure(figsize=(18, 6))
# sns.boxplot("salary", "specialisation", data=data)
### DOES MBA PERCENTAGES AFFECT PLACEMENT ###
# plt.figure(figsize=(18, 6))
# sns.boxplot("mba_p", "status", data=data)
# sns.lineplot("mba_p", "salary", data=data)
# plt.show()
### DATA PRE-PROCESSING ###
data.drop(['ssc_b', 'hsc_b'], axis=1, inplace=True)
### FEATURE ENCODING ###
# print(data.dtypes)
data['gender'] = data.gender.map({'M': 0, 'F': 1})
data['hsc_s'] = data.hsc_s.map({'Commence': 0, 'Science': 1, 'Arts': 2})
data['degree_t'] = data.degree_t.map({'Comm&Mgmt': 0, 'Sci&Tech': 1, 'Others': 2})
data['workex'] = data.workex.map({'No': 0, 'Yes': 1})
data['status'] = data.status.map({'Not Placed': 0, 'Placed': 1})
data['specialisation'] = data.specialisation.map({'Mkt&HR': 0, 'Mkt&Fin': 1})

# make a copy of the data
data_clf = data.copy()
data_reg = data.copy()

### Dropping Salary Feature ###
# separating features and target
# X = data_clf[['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation',
# 'mba_p']]
# fix error that there was NaN and infinity values present
# X = X.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
# Y = data_clf['status']
# train test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# dtree = DecisionTreeClassifier(criterion='entropy')
# dtree.fit(X_train, Y_train)
# y_pred = dtree.predict(X_test)
# print(accuracy_score(Y_test, y_pred))
# print(classification_report(Y_test, y_pred))
### Using Random Forest Algorithm ###
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# y_pred = random_forest.predict(X_test)
# print(accuracy_score(Y_test, y_pred))
# print(classification_report(Y_test, y_pred))
### FEATURE IMPORTANCE(PERCENTAGE) ###
# rows = list(X.columns)
# imp = pd.DataFrame(np.zeros(6*len(rows)).reshape(2*len(rows), 3))
# imp.columns = ['Classifier', 'Feature', 'Importance']
# Add Rows
# for index in range(0, 2*len(rows), 2):
# imp.iloc[index] = ['DecisionTree', rows[index//2], (100*dtree.feature_importances_[index//2])]
# imp.iloc[index + 1] = ['RandomForest', rows[index//2], (100*random_forest.feature_importances_[index//2])]
# plt.figure(figsize=(15, 5))
# sns.barplot('Feature', 'Importance', hue='Classifier', data=imp)
# plt.title('Computed Feature Importance')
# plt.show()

### Binary Classification with Logic Regression ###
# One Hot Encoding = encoding the categorical featured

X = data_clf[['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation',
              'mba_p']]
# fix error that there was NaN and infinity values present
X = X.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
Y = data_clf['status']
# reverse mapping from previous
X['gender'] = pd.Categorical(X.gender.map({0: 'M', 1: 'F'}))
X['hsc_s'] = pd.Categorical(X.hsc_s.map({0: 'Commerce', 1: 'Science', 2: 'Arts'}))
X['degree_t'] = pd.Categorical(X.degree_t.map({0: 'Comm&Mgmt', 1: 'Sci&Tech', 2: 'Others'}))
X['workex'] = pd.Categorical(X.workex.map({0: 'No', 1: 'Yes'}))
X['specialisation'] = pd.Categorical(X.specialisation.map({0: 'Mkt&HR', 1: 'Mkt&Fin'}))
X = pd.get_dummies(X)
column_names = X.columns.tolist()
# using the scaling to be sure salary doesn't skew data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# test train and split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3)

# the logistic regression model
logistic_reg = LogisticRegression()
# error fixed with () object with empty parenthesis which simply means default constructor.
# Started looking for other parameter
logistic_reg.fit(X_train, Y_train)
y_pred = logistic_reg.predict(X_test)

# print(accuracy_score(Y_test, y_pred))
# print(classification_report(Y_test, y_pred))

### Computing Feature Importance by Mean Decrease Accuracy (MDA) ###
perm = PermutationImportance(logistic_reg).fit(X_test, Y_test)
# print(eli5.format_as_text(eli5.explain_weights(perm)))
### had to use this method for not working in the right environment in notebook = eli5.show_weights(perm)
# plt.figure(figsize=(30, 10))
# plt.bar(column_names, perm.feature_importances_std_*100)
# plt.show()
### DATA PREPROCESSING ###
data_reg.dropna(inplace=True)
data_reg.drop('status', axis=1, inplace=True)
# print(data_reg.head())
### separate the dependent and independent variables
Y = data_reg['salary']
X = data_reg.drop('salary', axis=1)
column_names = X.columns.values
### scaling between 0-1 normalization
X_scaled = MinMaxScaler().fit_transform(X)
### Identifying and removing the outliers ###
# sns.kdeplot(Y)
# plt.show()
### Selecting the outliers###
# print(Y[Y > 400000])
### removing this from the data
X_scaled = X_scaled[Y < 400000]
Y = Y[Y < 400000]
# sns.kdeplot(Y)
# plt.show()
### Determining Least Significant Variable by R2 Score ###
linreg = LinearRegression()
sfs = SFS(linreg, k_features=1, forward=False, scoring='r2', cv=10)
sfs = sfs.fit(X_scaled, Y)
# fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
# plt.title('Sequential Backward Elimination')
# plt.grid()
# plt.show()
### See top 5 most significant ###
top_n = 5
# print(sfs.get_metric_dict()[top_n])
### Top names of the features with the highest R2 score ###
top_n_indices = list(sfs.get_metric_dict()[top_n]['feature_idx'])
# print(f'Most Significant{top_n} Features:')
# for col in column_names[top_n_indices]:
# print(col)

### SELECT THESE FEATURES ONLY ###
X_selected = X_scaled[:, top_n_indices]
lin_reg = LinearRegression()
lin_reg.fit(X_selected, Y)
# y_pred = lin_reg.predict(X_selected)
# print(f'R2 Score: {r2_score(Y, y_pred)}')
# print(f'MAE: {mean_absolute_error(Y, y_pred)}')

### DETERMINING THE LEAST SIGNIFICANT FEATURE BY P-VALUE###
### converting the column names to dataframe for readability
X_scaled = pd.DataFrame(X_scaled, columns=column_names)
Y = Y.values
### adding constants (1) for intercept before the line regression with statmodel
X_scaled = sm.add_constant(X_scaled)
# print(X_scaled.head())
### step 1 checking the p-value of all the features
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing etest_p from features due to highest p-value
X_scaled = X_scaled.drop('etest_p', axis=1)
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing workex from features due to highest p-value
X_scaled = X_scaled.drop('workex', axis=1)
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing specialisation from features due to highest p-value
X_scaled = X_scaled.drop('specialisation', axis=1)
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing ssc_p from features due to highest p-value
X_scaled = X_scaled.drop('ssc_p', axis=1)
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing hsc_p from features due to highest p-value
X_scaled = X_scaled.drop('hsc_p', axis=1)
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing hsc_s from features due to highest p-value
X_scaled = X_scaled.drop('hsc_s', axis=1)
# model = sm.OLS(Y, X_scaled)
# results = model.fit()
# print(results.summary())

### removing mba_p from features due to highest p-value
X_scaled = X_scaled.drop('mba_p', axis=1)
model = sm.OLS(Y, X_scaled)
results = model.fit()
print(results.summary())

### COMPLETED ###
