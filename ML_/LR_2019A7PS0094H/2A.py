import csv
import pandas as pd
from math import exp
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings( "ignore" )
data=pd.read_csv("dataset_LR.csv")
total_itr=1
flag=1
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
def loss(X,W,Y):
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
for q in range(1):
	avg_loss=avg_acc=avg_prec=avg_recall=avg_score=0
	test_acc=[]
	if q==0:
		rate=0.001		
	print("original")	
	for g in range(total_itr):
		train_data=data.sample(frac=0.7,random_state=1).reset_index(drop=True)	
		test_data =data.drop(train_data.index).reset_index(drop=True)
		train_data = train_data.reset_index(drop=True)
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
			k=1
			for x,y in zip(X_train,Y_train):
				flag=1
				w=update(x,y,w)
				if flag==1:
					k=1
				else:
					k=0	
			if (i%50) == 0:
				l=loss(X_train,w,Y_train)
				total_loss.append(-1*l)
				acc.append(accuracy(X_train,w,Y_train))
				x_values.append(i)			
			if k==0:
				break		
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
		avg_acc+=accu
		avg_prec+=precision
		avg_recall+=recall
		avg_score+=fscore			
		test_acc.append((accu*100))	
	print("Average loss: {}".format(avg_loss/1))	
	print("Average accuracy: {}%".format(avg_acc*100))
	print("Average precision: {}%".format(avg_prec*100))
	print("Average recall: {}%".format(avg_recall*100))
	print("Average F1 Score: {}%".format(avg_score*100))		
temp=data.iloc[:,0:1]
temp+=temp/20
data['attr1']=temp	
print("5per. increase in att1")
for q in range(1):
	avg_loss=avg_acc=avg_prec=avg_recall=avg_score=0
	test_acc=[]
	if q==0:
		rate=0.001		
	for g in range(total_itr):
		train_data=data.sample(frac=0.7, random_state=1).reset_index(drop=True)	
		test_data =data.drop(train_data.index).reset_index(drop=True)
		train_data = train_data.reset_index(drop=True)
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
			k=1
			for x,y in zip(X_train,Y_train):
				flag=1
				w=update(x,y,w)
				if flag==1:
					k=1
				else:
					k=0	
			if (i%50) == 0:
				l=loss(X_train,w,Y_train)
				total_loss.append(-1*l)
				acc.append(accuracy(X_train,w,Y_train))
				x_values.append(i)			
			if k==0:
				break		
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
		avg_acc+=accu
		avg_prec+=precision
		avg_recall+=recall
		avg_score+=fscore			
		test_acc.append((accu*100))
	print("Average loss: {}".format(avg_loss/1))	
	print("Average accuracy: {}%".format(avg_acc*100))
	print("Average precision: {}%".format(avg_prec*100))
	print("Average recall: {}%".format(avg_recall*100))
	print("Average F1 Score: {}%".format(avg_score*100))	
data=pd.read_csv("dataset_LR.csv")	
temp=data.iloc[:,1:2]
temp+=temp/20
data['attr2']=temp	
print("5per. increase in att2")	
for q in range(1):
	avg_loss=avg_acc=avg_prec=avg_recall=avg_score=0
	test_acc=[]
	if q==0:
		rate=0.001		
	for g in range(total_itr):
		train_data=data.sample(frac=0.7, random_state=1).reset_index(drop=True)	
		test_data =data.drop(train_data.index).reset_index(drop=True)
		train_data = train_data.reset_index(drop=True)
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
			k=1
			for x,y in zip(X_train,Y_train):
				flag=1
				w=update(x,y,w)
				if flag==1:
					k=1
				else:
					k=0	
			if (i%50) == 0:
				l=loss(X_train,w,Y_train)
				total_loss.append(-1*l)
				acc.append(accuracy(X_train,w,Y_train))
				x_values.append(i)			
			if k==0:
				break		
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
		avg_acc+=accu
		avg_prec+=precision
		avg_recall+=recall
		avg_score+=fscore			
		test_acc.append((accu*100))
	print("Average loss: {}".format(avg_loss/1))	
	print("Average accuracy: {}%".format(avg_acc*100))
	print("Average precision: {}%".format(avg_prec*100))
	print("Average recall: {}%".format(avg_recall*100))
	print("Average F1 Score: {}%".format(avg_score*100))		
data=pd.read_csv("dataset_LR.csv")	
temp=data.iloc[:,2:3]
temp+=temp/20
data['attr3']=temp	
print("5per. increase in att3")		
for q in range(1):
	avg_loss=avg_acc=avg_prec=avg_recall=avg_score=0
	test_acc=[]
	if q==0:
		rate=0.001		
	for g in range(total_itr):
		train_data=data.sample(frac=0.7, random_state=1).reset_index(drop=True)	
		test_data =data.drop(train_data.index).reset_index(drop=True)
		train_data = train_data.reset_index(drop=True)
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
			k=1
			for x,y in zip(X_train,Y_train):
				flag=1
				w=update(x,y,w)
				if flag==1:
					k=1
				else:
					k=0	
			if (i%50) == 0:
				l=loss(X_train,w,Y_train)
				total_loss.append(-1*l)
				acc.append(accuracy(X_train,w,Y_train))
				x_values.append(i)			
			if k==0:
				break		
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
		avg_acc+=accu
		avg_prec+=precision
		avg_recall+=recall
		avg_score+=fscore			
		test_acc.append((accu*100))
	print("Average loss: {}".format(avg_loss/1))	
	print("Average accuracy: {}%".format(avg_acc*100))
	print("Average precision: {}%".format(avg_prec*100))
	print("Average recall: {}%".format(avg_recall*100))
	print("Average F1 Score: {}%".format(avg_score*100))
data=pd.read_csv("dataset_LR.csv")	
temp=data.iloc[:,3:4]
temp+=temp/20
data['attr4']=temp	
print("5per. increase in att4")		
for q in range(1):
	avg_loss=avg_acc=avg_prec=avg_recall=avg_score=0
	test_acc=[]
	if q==0:
		rate=0.001		
	for g in range(total_itr):
		train_data=data.sample(frac=0.7, random_state=1).reset_index(drop=True)	
		test_data =data.drop(train_data.index).reset_index(drop=True)
		train_data = train_data.reset_index(drop=True)
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
			k=1
			for x,y in zip(X_train,Y_train):
				flag=1
				w=update(x,y,w)
				if flag==1:
					k=1
				else:
					k=0	
			if (i%50) == 0:
				l=loss(X_train,w,Y_train)
				total_loss.append(-1*l)
				acc.append(accuracy(X_train,w,Y_train))
				x_values.append(i)			
			if k==0:
				break		
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
		avg_acc+=accu
		avg_prec+=precision
		avg_recall+=recall
		avg_score+=fscore			
		test_acc.append((accu*100))
	print("Average loss: {}".format(avg_loss/1))	
	print("Average accuracy: {}%".format(avg_acc*100))
	print("Average precision: {}%".format(avg_prec*100))
	print("Average recall: {}%".format(avg_recall*100))
	print("Average F1 Score: {}%".format(avg_score*100))	