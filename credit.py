import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify as sqy
from wordcloud import WordCloud

df = pd.read_csv("german_credit_data.csv")
df.head()

df.shape
df.columns

df.info()

df.describe()

total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

sns.countplot('Risk', data=df)
plt.title('Risk Distribution', fontsize=14)
plt.show()

plt.figure(figsize =(20,20))
Corr=df[df.columns].corr()
sns.heatmap(Corr,annot=True)

# now let us check in the number of Percentage
Count_good_transaction = len(df[df["Risk"]=='good']) # good transaction are repersented by 0
Count_bad_transacaion = len(df[df["Risk"]=='bad']) # bad by 1
Percentage_of_good_transaction = Count_good_transaction/(Count_good_transaction+Count_bad_transaction)
print("percentage of good transaction is",Percentage_of_good_transaction*100)
Percentage_of_bad_transaction= Count_bad_transaction/(Count_good_transaction+Count_bad_transaction)
print("percentage of bad transaction is",Percentage_of_bad_transaction*100)

df = df.rename(columns={'Credit amount':'Credit'})

temp = df['Checking account'].value_counts()
plt.figure(figsize=(15,8))
sns.barplot(temp.index, temp.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('Range of the people', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Count of people status", fontsize=16)
plt.show()

good_transaction = df[df["Risk"]=='good']
bad_transaction= df[df["Risk"]=='bad']
plt.figure(figsize=(10,6))
plt.subplot(121)
good_transaction.Credit.plot.hist(title="Good Transaction")
plt.subplot(122)
bad_transaction.Credit.plot.hist(title="Bad Transaction")

plt.figure(figsize=(10,6))
plt.subplot(121)
good_transaction.Duration.plot.hist(title="Good Transaction")
plt.subplot(122)
bad_transaction.Duration.plot.hist(title="Bad Transaction")

good = df[df["Risk"]=='good']
bad = df[df["Risk"]=='bad']

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(bad.Duration, bad.Credit)
ax1.set_title('Fraud')
ax2.scatter(good.Duration, good.Credit)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Age of transaction vs Amount by class')
ax1.scatter(bad.Age, bad.Credit)
ax1.set_title('Bad')
ax2.scatter(good.Age, good.Credit)
ax2.set_title('Good')
plt.xlabel('Age (in years)')
plt.ylabel('Amount')
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Gender of transaction vs Amount by class')
ax1.scatter(bad.Sex, bad.Credit)
ax1.set_title('Bad')
ax2.scatter(good.Sex, good.Credit)
ax2.set_title('Good')
plt.xlabel('Gender')
plt.ylabel('Amount')
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Job vs Amount by class')
ax1.scatter(bad.Job, bad.Credit)
ax1.set_title('Bad')
ax2.scatter(good.Job, good.Credit)
ax2.set_title('Good')
plt.xlabel('Based on Job Grading')
plt.ylabel('Amount')
plt.show()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df["Risk"]= le.fit_transform(df["Risk"])
df["Sex"] = le.fit_transform(df["Sex"])
df["Housing"] = le.fit_transform(df["Housing"])

df["Saving accounts"] = df["Saving accounts"].replace({"little":0, "moderate":1, "rich":2, "quite rich":3})
df["Checking account"] = df["Checking account"].replace({"little":0, "moderate":1, "rich":2})
df = df.fillna(1)
df1 = df.drop(columns = 'Purpose')
df1.head()

plt.figure(figsize =(20,20))
Corr=df[df.columns].corr()
sns.heatmap(Corr,annot=True)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42

columns = df1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Risk"]]
# Store the variable we are predicting 
target = "Risk"
# Define a random state 
state = np.random.RandomState(42)
X = df1[columns]
Y = df1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)

outlier_fraction = len(bad)/float(len(good))
classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=42, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1, random_state=43)
   
}

n_outliers = len(bad)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 1 for good transactions , 0 for bad transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))
    
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y, y_pred)
print(cm)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
