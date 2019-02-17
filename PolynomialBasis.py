#creating a vandermode matrix for the data which we generated in linearRegression.
import numpy as np
def polynomial_transform(X, d):
    vander=[]
    for i in range(len(X)):
        new=[]
        for j in range(d):
            new.append(X[i]**j)
        vander.append(new)
    return np.flip(np.array(vander),axis=1)
    
#training the model    
import numpy as np
def train_model(Phi, y):
  Phit=Phi.transpose()
  Inner=np.dot(Phit,Phi)
  InvInner=np.linalg.inv(Inner)
  a=np.dot(InvInner,Phit)
  return np.dot(a,y)  
  
# Python function that evaluates the model using mean squared error.
import numpy as np
def evaluate_model(Phi, y, w):  
    mul=np.dot(Phi,w)
    res=(np.subtract(y,mul))
    res=np.power(res,2)
    res=np.sum(res)
    return res/len(y)
    
    
#explore the effect of complexity by varying  d=3,6,9,⋯,24 to steadily increase the non-linearity of the models.
w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models
for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])   


#visualize the models
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])
