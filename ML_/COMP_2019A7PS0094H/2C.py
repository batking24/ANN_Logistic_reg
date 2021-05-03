import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns
data=pd.read_csv("dataset_comb.csv")
data1=data.iloc[:,:-1].values
data2=data.iloc[:,-1:]
y_values=[]
names=['id','Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','EquivDiameter','Extent','Perimeter','Roundness','AspectRation']
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(data1)
data = pd.DataFrame(d,columns=names)
data1=data.iloc[:,:].values
for index,row in data2.iterrows():
	if row['Class']=='jasmine':
		row['Class']=1
		y_values.append(1)
	else:
		row['Class']=2
		y_values.append(2)		
data["Class"]=data2	
temp=data
data=data.values	
x=data[:,:-1]
y=data[:,-1:]	
seed=7
models = []
models.append(('MLP',MLPClassifier()))
models.append(('PR',Perceptron()))
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=7, random_state=seed,shuffle=True)
	cv_results = model_selection.cross_val_score(model, x, y_values, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
cor=temp.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True)
plt.show()