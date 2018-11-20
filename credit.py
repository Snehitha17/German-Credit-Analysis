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

temp = df['Purpose'].value_counts()
plt.figure(figsize=(15,8))
sns.barplot(temp.index, temp.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('Usage items by people', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Daily uses of people ", fontsize=16)
plt.show()

temp = df['Duration'].value_counts()
plt.figure(figsize=(15,8))
sns.barplot(temp.index, temp.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('Timings of the credit ', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Basedon duration of people ", fontsize=16)
plt.show()

