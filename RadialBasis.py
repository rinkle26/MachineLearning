#computing a radial-basis kernel for the synthetic dataset.
import numpy as np
def radial_basis_transform(X, B, gamma=0.1):
    res=[]
    for i in range(len(X)):
        new=[]
        for j in range(len(B)): 
            new.append(-gamma*((X[i]-B[j])**2))
        res.append(new)
    return np.exp(res)
 
#training the model 
#Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
import numpy as np
def train_ridge_model(Phi, y, lam):
    a1=np.dot(Phi.transpose(),Phi)
    a2=(lam*np.identity(len(Phi.transpose()))) 
    a3=np.add(a1,a2)
    res=np.linalg.inv(a3) 
    return np.dot(res,np.dot(Phi.transpose(),y))

#evaluating the performance on the transformed validation and test data.
w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models
lam=10**-3
while lam<=10**3:  # Iterate over polynomial degree
    Phi_trn = radial_basis_transform(X_trn,X_trn)                 # Transform training data into d dimensions
    w[lam] = train_ridge_model(Phi_trn, y_trn,lam)                       # Learn model on training data
    
    Phi_val = radial_basis_transform(X_val,X_trn)                 # Transform validation data into d dimensions
    validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam])  # Evaluate model on validation data
    
    Phi_tst = radial_basis_transform(X_tst, X_trn)           # Transform test data into d dimensions
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam])  # Evaluate model on test data
    lam=lam*10

# Plot all the models -both learned and true
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.xscale("log")    

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
lam=10**-3
this_list=list()
while lam<=10**3:
  this_list.append(lam)
  X_lam = radial_basis_transform(x_true, X_trn)
  y_lam = X_lam @ w[lam]
  #print(w[lam])
  plt.plot(x_true, y_lam, marker='None', linewidth=2)
  lam=lam*10
plt.legend(['true'] + this_list)
plt.axis([-8, 8, -15, 15])

#As lambda increases, linearity of model increases and vice versa.
#minimum error is on lambda=10^-3
#validationErr- minimum 46.160 on lambda=10^-3 
#testErr- minimum 37.50 on lambda=10^-3 
