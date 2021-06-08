import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import association_metrics as am
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
pd.set_option('display.max_colwidth', 1000)  # or 199

dataset = pd.read_csv("AccidentsVrilissiaDataset.csv", keep_default_na=False)
dataset = dataset.drop(['Source'], axis=1)

def preprocess():
	for index in dataset.index:
		dataset['Date'][index] = datetime.strptime(dataset['Date'][index],'%d/%m/%Y')

	dataset['Day'] = dataset['Date'].map(lambda x: x.day)
	dataset['Weekday'] = dataset['Date'].apply(lambda x: x.weekday())
	dataset['Month'] = dataset['Date'].map(lambda x: x.month)
	dataset['Year'] = dataset['Date'].map(lambda x: x.year)

	dataset['Severity'] = dataset['Severity'].astype("category")
	dataset['Cause'] = dataset['Cause'].astype("category")
	dataset['Weekday'] = dataset['Weekday'].astype("category")
	dataset['Year'] = dataset['Year'].astype("category")
	dataset['Month'] = dataset['Month'].astype("category")
	dataset['Point of accident'] = dataset['Point of accident'].astype("category")
	dataset.drop(['Date'], axis = 1)

def plotYears():
	years = dataset['Year'].unique()
	accidentsPerYear = dataset.groupby(['Year']).size().reset_index(name='No. of accidents')
	print(accidentsPerYear)
	plt.title('Recorded accidents by Year', fontweight='bold')
	plt.bar(accidentsPerYear['Year'], accidentsPerYear['No. of accidents'])
	plt.tight_layout()
	plt.show()

def plotCause():
	cause = dataset['Cause'].unique()
	accidentsPerCause = dataset.groupby(['Cause']).size().reset_index(name='No. of accidents')
	print(accidentsPerCause)
	colors = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:red', 'tab:pink', 'tab:purple', 'yellow', 'tab:brown', 'tab:green', 'tab:olive', 'tab:gray']
	plt.title('Recorded accidents by Cause', fontweight='bold')
	plt.pie(accidentsPerCause['No. of accidents'], labels = accidentsPerCause['Cause'], autopct='%1.2f%%', labeldistance=1.1, colors = colors)
	plt.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
	plt.tight_layout()
	plt.show()

def plotSeverity():
	cause = dataset['Cause'].unique()
	accidentsPerSeverity = dataset.groupby(['Severity']).size().reset_index(name='No. of accidents')
	print(accidentsPerSeverity)
	colors = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:red', 'tab:pink', 'tab:purple', 'yellow', 'tab:brown', 'tab:green', 'tab:olive', 'tab:gray']
	plt.title('Recorded accidents by Severity', fontweight='bold')
	plt.pie(accidentsPerSeverity['No. of accidents'], labels = accidentsPerSeverity['Severity'], autopct='%1.2f%%', labeldistance=1.1, colors = colors)
	plt.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
	plt.tight_layout()
	plt.show()

def plotWeekDay():
	cause = dataset['Weekday'].unique()
	accidentsPerWeekday = dataset.groupby(['Weekday']).size().reset_index(name='No. of accidents')
	print(accidentsPerWeekday)
	plt.title('Recorded accidents by Weekday', fontweight='bold')
	plt.bar(accidentsPerWeekday['Weekday'], accidentsPerWeekday['No. of accidents'])
	plt.xticks([0,1,2,3,4,5,6],['Monday', 'Tuesday','Wednesday', 'Thursady', 'Friday', 'Saturday', 'Sunday'])
	plt.tight_layout()
	plt.show()

def plotMonth():
	cause = dataset['Month'].unique()
	accidentsPerMonth = dataset.groupby(['Month']).size().reset_index(name='No. of accidents')
	print(accidentsPerMonth)
	plt.title('Recorded accidents by Month', fontweight='bold')
	plt.bar(accidentsPerMonth['Month'], accidentsPerMonth['No. of accidents'])
	plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],['January', 'February','March', 'April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December'])
	plt.tight_layout()
	plt.show()

def plotPointOfAccident():
	points = dataset['Point of accident'].unique()
	accidentsPerPoint = dataset.groupby(['Point of accident']).size().reset_index(name='No. of accidents')
	accidentsPerPoint['Point of accident'] = accidentsPerPoint['Point of accident'].astype("category")
	accidentsPerPoint = accidentsPerPoint.sort_values(by=['No. of accidents'], ascending=False)
	print(accidentsPerPoint)
	plt.title('Recorded accidents by Point', fontweight='bold')
	sns.barplot(y = accidentsPerPoint['Point of accident'].unique(), x = accidentsPerPoint['No. of accidents'])
	plt.yticks(accidentsPerPoint['Point of accident'].unique(), fontsize='7')
	#plt.yticklabels(accidentsPerPoint['Point of accident'].unique())
	plt.xlabel('# of accidents', fontsize=10)
	plt.ylabel('Point', fontsize=10)
	plt.tight_layout()
	plt.show()

def plotPointOfAccidentLgOne():
	points = dataset['Point of accident'].unique()
	accidentsPerPoint = dataset.groupby(['Point of accident']).size().reset_index(name='No. of accidents')
	accidentsPerPoint = accidentsPerPoint[accidentsPerPoint['No. of accidents'] > 1]
	accidentsPerPoint['Point of accident'] = accidentsPerPoint['Point of accident'].astype("category")
	accidentsPerPoint = accidentsPerPoint.sort_values(by=['No. of accidents'], ascending=False)
	print(accidentsPerPoint)
	plt.title('Recorded accidents by Point', fontweight='bold')
	sns.barplot(y = accidentsPerPoint['Point of accident'].unique(), x = accidentsPerPoint['No. of accidents'])
	plt.yticks(np.arange(accidentsPerPoint['Point of accident'].count()),accidentsPerPoint['Point of accident'].unique(), fontsize=6)
	#plt.yticklabels(accidentsPerPoint['Point of accident'].unique())
	plt.xlabel('# of accidents', fontsize=10)
	plt.ylabel('Point', fontsize=10)
	plt.show()

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def correlation():
	cramersv = am.CramersV(dataset) 
	conf_matrix = cramersv.fit()
	print(conf_matrix)
	plt.title('Cramers V correlation', fontweight='bold')
	sns.heatmap(conf_matrix, cmap='coolwarm')
	plt.xticks(rotation=360)
	plt.show()


def train_random_forest():
	le = preprocessing.LabelEncoder()
	le.fit(dataset['Cause'])
	dataset['Cause'] = le.transform(dataset['Cause'])

	le.fit(dataset['Severity'])
	dataset['Severity'] = le.transform(dataset['Severity'])
	print(le.classes_)

	X = dataset[['Cause', 'Weekday', 'Year', 'Month', 'Point of accident']]
	y = dataset['Severity']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	clf=RandomForestClassifier(n_estimators=100)

	#Train the model using the training sets y_pred=clf.predict(X_test)
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def train_MLP():
	le = preprocessing.LabelEncoder()
	le.fit(dataset['Cause'])
	dataset['Cause'] = le.transform(dataset['Cause'])

	le.fit(dataset['Severity'])
	dataset['Severity'] = le.transform(dataset['Severity'])
	print(le.classes_)

	X = dataset[['Cause', 'Weekday', 'Year', 'Month', 'Point of accident']]
	y = dataset['Severity']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	mlp = MLPClassifier(hidden_layer_sizes=(4, 3), max_iter=1500,activation = 'relu',solver='adam',random_state=1,verbose=True)

	#Train the model using the training sets y_pred=clf.predict(X_test)
	mlp.fit(X_train,y_train)
	y_pred=mlp.predict(X_test)
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	
preprocess()
train_MLP()
