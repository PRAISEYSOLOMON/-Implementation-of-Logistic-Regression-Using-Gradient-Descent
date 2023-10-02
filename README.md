# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import the packages required.
 2. Read the dataset.
 3. Define X and Y array.
 4. Define a function for costFunction,cost and gradient.
 5. Define a function to plot the decision boundary and predict the Regression value.
 

## Program:
```
/*
# Developed by: PRAISEY S
# RegisterNumber:  212222040117
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
print("Array of X") 
X[:5]
print("Array of y") 
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
print("Prediction value of mean")
np.mean(predict(res.x,X)==y)

*/
```

## Output:

![out 5 1](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/1474ebfe-9892-46b1-9178-4beb6479853f)

![out 5 2](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/affc1250-1649-415a-b2ce-097b11703864)

![out 5 3](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/8f533d92-4474-4ccf-b8fe-0f727bd8bb70)

![out 5 4](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/d988b9e1-397a-441d-9dc6-214fe6217c8b)

![out 5 5](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/15b4324c-8b69-4486-a795-159643350efe)

![out 5 6](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/715d451b-5cb8-4da8-905a-c39f6fcfd339)

![out 5 7](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/72af946e-95bd-4d60-abdf-880ab25f167f)

![out 5 8](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/272e9a10-5ce2-4379-95de-f94c8bc3d3f8)

![out 5 9](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/9fc8aa10-1e27-4578-8894-c7205b958d73)

![out 5 10](https://github.com/PRAISEYSOLOMON/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394259/ed361c17-561e-4240-b130-74ddca66f48f)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

