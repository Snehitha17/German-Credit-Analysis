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

