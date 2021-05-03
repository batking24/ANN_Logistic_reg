import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from math import exp,log
warnings.filterwarnings( "ignore" )
np.random.seed(1)
data=pd.read_csv("dataset_NN.csv")
data1=data.iloc[:,:-1]
data2=data.iloc[:,-1:]
data=(data1-data1.min())/(data1.max()-data1.min())
data["class"]=data2
train_data=data.sample(frac=0.7).reset_index(drop=True)	
test_data =data.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
hidden_nodes=8
output_nodes=10
attributes=train_data.shape[1]-1
wh=np.random.randn(attributes,hidden_nodes)
bh=np.random.randn(1,hidden_nodes)
wo=np.random.randn(hidden_nodes,output_nodes)
bo=np.random.randn(1,output_nodes)
rows=train_data.shape[0]
rate=0.0001
loss_list=[]
x_values=[]
accuracy=[]
def relu(x):
	return np.maximum(x, 0)
def der_relu(x):
	x[x<=0] = 0
	x[x>0] = 1
	return x	
def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)
xi=np.array(train_data.iloc[:,:-1].values)
value=train_data.iloc[:,-1:].values
output=np.zeros((rows,output_nodes))
for i in range(rows):
	output[i, value[i][0]-1] = 1	
for itr in range(50000):
	zh=np.dot(xi,wh)+bh
	ah=relu(zh)
	zo=np.dot(ah,wo)+bo	
	ao=softmax(zo)
	dcost_dzo=ao-output
	dcost_dwo=np.dot(ah.T,dcost_dzo)
	wo-=rate*dcost_dwo
	bo-=rate*dcost_dzo.sum(axis=0)
	dcost_dah=np.dot(dcost_dzo,wo.T)
	dcost_dzh=dcost_dah*der_relu(zh)
	dcost_dwh=np.dot(xi.T,dcost_dzh)
	wh-=rate*dcost_dwh
	bh-=rate*dcost_dzh.sum(axis=0)
	if itr % 1000== 0:
		loss = np.sum(-output * np.log(ao))
		loss/=rows
		print('Loss function value: ', loss)
		loss_list.append(loss)
		x_values.append(itr)
		count=0
		for k in range(rows):
			if np.argmax(output[k])==np.argmax(ao[k]):
				count+=1
		accuracy.append((count*100)/rows)		
		print((count*100)/rows)	
plt.scatter(x_values,loss_list)	
plt.title("Loss")
plt.show()		
plt.scatter(x_values,accuracy)	
plt.title("Accuracy")
plt.show()
rows=test_data.shape[0]	
xi=np.array(test_data.iloc[:,:-1].values)
value=test_data.iloc[:,-1:].values
output=np.zeros((rows,output_nodes))	
for i in range(rows):
	output[i, value[i][0]-1] = 1
zh=np.dot(xi,wh)+bh
ah=relu(zh)
zo=np.dot(ah,wo)+bo	
ao=softmax(zo)	
count=0
for k in range(rows):
	if np.argmax(output[k])==np.argmax(ao[k]):
		count+=1	
print((count*100)/rows)				
'''
d=train_data.sample()
inp=np.array(d.iloc[:,:-1].values)
value=d.iloc[:,-1:].values
output=np.zeros(output_nodes)
output[value[0]-1]=1
	#forward
zh=np.dot(inp,wh)+bh   #1*8
ah=leaky_relu(zh)   #1*8
zo=np.dot(ah,wo)+bo  # 1*10
ao=softmax(zo)	
print(output)
print(ao)	
plt.scatter(x_values,loss_list)	
plt.title("Loss")
plt.show()	'''