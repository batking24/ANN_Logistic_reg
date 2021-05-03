import csv
import pandas as pd
from math import exp
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings( "ignore" )
data=pd.read_csv("dataset_LR.csv")
total_itr=10
flag=1

#It is a special case of linear regression where the target variable is categorical in nature. It uses a log of odds as the dependent variable. Logistic Regression predicts the probability of occurrence of a binary event utilizing a logit function.
# gradient decent:
# we train our model in whole training data( say 70 % of total)
# stochastic gradient decent:
# we take a perecentage of training data( th

# mini batch gradient decent:
# we take a batch of data for training data


def update(X,Y,W):
	A = 1 / ( 1 + np.exp( - (np.dot(X,W[1:])+W[0])))
	tmp=A-Y
	dw=x*tmp
	if tmp==0:
		flag=0
	db=tmp
	W[1:]=W[1:]-rate*dw
	W[0]-=db
	return W
def predict(X,W):
	Z = 1 / ( 1 + np.exp( - ( np.dot(X,W[1:])+W[0]) ) )       
	Y = np.where( Z > 0.5, 1, 0)
	return Y	
def loss(X,W,Y):#t log(y)+(1-t) log(1-y)
	l=0
	for x,y in zip(X,Y):
		temp = 1 / ( 1 + np.exp( - (np.dot(x,W[1:])+W[0])))
		if temp!=0:
			l+=y*math.log10(temp)
		if 1-temp!=0:	
			l+=(1-y)*math.log10(1-temp)
	return l
def accuracy(X,W,Y):
	a=0
	tp=fp=tn=fn=0			
	for x,y in zip(X,Y):
		temp=predict(x,W)
		if temp==1 and y==1:
			tp+=1
		elif temp==1 and y==0:
			fp+=1
		elif temp==0 and y==1:
			fn+=1
		else:
			tn+=1
	a=100*(tp+tn)/(tp+tn+fp+fn)		
	return a		
for q in range(3):
	avg_loss=avg_acc=avg_prec=avg_recall=avg_score=0
	test_acc=[]
	if q==0:
		rate=0.001
	elif q==1:
		rate=0.01
	else:
		rate=0.1		
	for g in range(total_itr):
		train_data=data.sample(frac=0.7).reset_index(drop=True)	
		test_data =data.drop(train_data.index).reset_index(drop=True)
		train_data = train_data.reset_index(drop=True)
		print("For {} split:".format(g+1))
		print("Train data")
		print(train_data)
		print("Test data")
		print(test_data)
		total_loss=[]
		x_values=[]
		acc=[]
		X_train=train_data.iloc[:,0:4].values
		Y_train=train_data.iloc[:,-1:].values
		X_test=test_data.iloc[:,0:4].values
		Y_test=test_data.iloc[:,-1:].values
		w=np.zeros(train_data.shape[1])
		iterations=1000
		m,n=train_data.shape
		for i in range(iterations):
			d=train_data.sample()
			x=d.iloc[:,0:4].values
			y=d.iloc[:,-1:].values
			w=update(x,y,w)
			if (i%50) == 0:
				l=loss(X_train,w,Y_train)
				total_loss.append(-1*l)
				acc.append(accuracy(X_train,w,Y_train))
				x_values.append(i)					
		print(w)
		tp=fp=tn=fn=0			
		for x,y in zip(X_train,Y_train):
			temp=predict(x,w)
			if temp==1 and y==1:
				tp+=1
			elif temp==1 and y==0:
				fp+=1
			elif temp==0 and y==1:
				fn+=1
			else:
				tn+=1	
		accu=(tp+tn)/(tp+tn+fp+fn)
		precision=tp/(tp+fp)
		recall=tp/(tp+fn)
		fscore=(2*precision*recall)/(precision+recall)	
		print("For Training data:")			
		print("Accuracy: {}%".format(accu*100))
		print("Precison: {}%".format(precision*100))
		print("Recall: {}%".format(recall*100))
		print("F1 Score: {}%".format(fscore*100))
		tp=fp=tn=fn=0			
		for x,y in zip(X_test,Y_test):
			temp=predict(x,w)
			if temp==1 and y==1:
				tp+=1
			elif temp==1 and y==0:
				fp+=1
			elif temp==0 and y==1:
				fn+=1
			else:
				tn+=1	
		accu=(tp+tn)/(tp+tn+fp+fn)
		l=loss(X_test,w,Y_test)
		avg_loss+=l
		precision=tp/(tp+fp)
		recall=tp/(tp+fn)
		fscore=(2*precision*recall)/(precision+recall)	
		print("For Testing data:")
		avg_acc+=accu
		avg_prec+=precision
		avg_recall+=recall
		avg_score+=fscore			
		test_acc.append((accu*100))
		print("Accuracy: {}%".format(accu*100))
		print("Precison: {}%".format(precision*100))
		print("Recall: {}%".format(recall*100))
		print("F1 Score: {}%".format(fscore*100))
		plt.scatter(x_values,total_loss)	
		plt.title("Loss on training data")
		plt.show()		
		plt.scatter(x_values,acc)
		plt.title("Accuracy over training data")
		plt.show()
	print("For learning rate: {}".format(rate))	
	print("Average loss: {}".format(avg_loss/10))	
	print("Average accuracy: {}%".format(avg_acc*10))
	print("Average precision: {}%".format(avg_prec*10))
	print("Average recall: {}%".format(avg_recall*10))
	print("Average F1 Score: {}%".format(avg_score*10))	
	x=np.linspace(0,10,10)	
	plt.scatter(x,test_acc)	
	plt.title("Accuracy")
	plt.show()		