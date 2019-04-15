
"""
Created on Sun Apr 14 17:31:00 2019

@author: rinkleseth
"""


import numpy as np
import os
import graphviz
import operator
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    
    """
    indices=dict()
    for i in range(len(x)):
        if x[i] in indices:
            indices[x[i]].append(i)
        else:
            indices.setdefault(x[i], [])
            indices[x[i]].append(i)
            
    return indices


def entropy(y,w=None):
    #function to calculate entropy.
    dict_y=partition(y)
    prob_sum=0
    total_weight_sum=np.sum(w)    
    
    for key in dict_y:
        key_weight_sum=0
        for indice in dict_y[key]:
            key_weight_sum+=w[indice]
        
        prob=-key_weight_sum/total_weight_sum*np.log2(key_weight_sum/total_weight_sum)
        prob_sum+=prob  
        
    return(prob_sum)


def mutual_information(x, y,w=None):
   #function to calculate mutual information of attributes,returns a dictionary of mutual informations for all attributes.
    dict_x=partition(x)
 
    entropy_y=entropy(y,w=w)
 
    ans=0
    total_weight=np.sum(w)
    
    for key in dict_x:
        y_ans=[]
        w_ans=[]
        for i in dict_x[key]:
            y_ans.append(y[i])
            w_ans.append(w[i])
        
        entropy_x=entropy(y_ans,w=w_ans)
        
         
        prob=np.sum(w_ans)/total_weight
        ans+=prob*entropy_x
        
    mi=entropy_y-ans
    return mi


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=3,w=None):
   #function to create decision tree recursively.
    dtree=dict()
      #----------------------------------------------------terminate
    if x.size==0:
        return dtree
    if depth>=max_depth:
        counts=np.bincount(y)
        return np.argmax(counts) 

    if attribute_value_pairs!=None and len(attribute_value_pairs)==0:
        counts=np.bincount(y)
        return np.argmax(counts)
    
    if len(np.unique(y))==1:
        return y[0]
    
    #----------------------------------------------------attribute-value-pairs
    if attribute_value_pairs==None:
            attribute_value_pairs=[]
            n=len(x[0])
            for i in range(n):
                col=x[:,i]
                keys=np.unique(col)
                for key in keys:
                    attribute_value_pairs.append((i,key))
     
            
    mi_dict=dict()
   #(a1,2)
    for i,value in attribute_value_pairs:
        #print(i,value)
        col=x[:,i] #each attribute column
        #print(col)
        bin_col=np.zeros((len(col),), dtype=int)
        
        for j in range(len(col)):
            if col[j]==value:
                bin_col[j]=1
        #print(bin_col)  
        mi=mutual_information(bin_col,y,w=w)
        mi_dict[(i,value)]=mi

    #----------------------------------------------------------
    
  
    attr,value=max(mi_dict.items(), key=operator.itemgetter(1))[0]
    best_col=x[:,attr]

    
    #-----------------------------------split on best attribute
    xsubset_true=[]
    xsubset_false=[]
    ysubset_true=[]
    ysubset_false=[]
    weight_true=[]
    weight_false=[]

    for i in range(len(x)):
        if best_col[i]==value:
            xsubset_true.append(x[i])
            ysubset_true.append(y[i])
            weight_true.append(w[i])
        else:
            xsubset_false.append(x[i])
            ysubset_false.append(y[i])
            weight_false.append(w[i])
            
    xsubset_true=np.asarray(xsubset_true)
    xsubset_false=np.asarray(xsubset_false)
    ysubset_true=np.asarray(ysubset_true)
    ysubset_false=np.asarray(ysubset_false)
    weight_true=np.asarray(weight_true)
    weight_false=np.asarray(weight_false)
    
    attribute_value_pairs_child=[]
    attribute_value_pairs_child=attribute_value_pairs.copy()
    attribute_value_pairs_child.remove((attr,value))
    
    #----------------------------------------recursive calls for subtree
    dtree.setdefault((attr,value,True),{})
    dtree.setdefault((attr,value,False),{})
    
    dtree[(attr,value,True)]=id3(xsubset_true,ysubset_true,attribute_value_pairs_child,depth+1,max_depth=max_depth,w=weight_true)
    dtree[(attr,value,False)]=id3(xsubset_false,ysubset_false,attribute_value_pairs_child,depth+1,max_depth=max_depth,w=weight_false)
 
    return dtree


def predict_example(x, h_ens):
    #predict output labels for the models learned
    weighted_avg=0
    sum_alpha=0
    for i in range(len(h_ens)):
        ypred=predict_example1(x,h_ens[i][0]) #predicted y values for a particular tree i.e 1/0
        weighted_avg+=ypred*h_ens[i][1]     #alpha*ypred
        sum_alpha+=h_ens[i][1]
    
    final_hypothesis=weighted_avg/sum_alpha
    
    if(final_hypothesis<0.5):
        return 0
    else:
        return 1
    


def predict_example1(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree

    """
    key=list(tree.keys())[0]
    (col,attr,val)=key

    if x[col]==attr:
      
        if type(tree[(col,attr,True)]) is dict:
            return predict_example1(x,tree[(col,attr,True)])

        else:

            return tree[(col,attr,True)]

    else:
   
        if type(tree[(col,attr,False)]) is dict:
            return predict_example1(x,tree[(col,attr,False)])

        else:
            return tree[(col,attr,False)]


def compute_error(y_true, y_pred,w=None):

     if w!=None:
        weighted_error_sum=0
   
        total_weighted_sum=np.sum(w)
        count=0
        for i in range(len(y_true)):
            
            if y_true[i]!=y_pred[i]:
                count+=1
                weighted_error_sum+=w[i]*1
    
        
        return float(weighted_error_sum/total_weighted_sum)
     else:
        n=len(y_pred)
        count=0
        for i in range(n):
            if y_true[i]!=y_pred[i]:
                count+=1
        
        return float(count)/float(n)

def boosting(x, y, max_depth, num_stumps):
    weight=[]
    dtree=dict()
    alpha=dict()
    h_ens=[]
    for i in range(len(y)):
        weight.append(1/len(y))
    for i in range(num_stumps):
        z=0
      
        dtree[i]=id3(x,y,max_depth=max_depth,w=weight)
        y_pred2 = [predict_example1(X, dtree[i]) for X in x] #predictions on training set

        error_t=compute_error(y,y_pred2,w=weight)
 
        alpha[i]=float(1/2)*math.log(float((1-error_t)/error_t))
        ensemble=[]
        ensemble.append(dtree[i])
        ensemble.append(alpha[i])
        h_ens.append(ensemble)

        
        for j in range(len(y)):    #normalization factor 
            prod=alpha[i]*y_pred2[j]*y[j]
            z+=weight[j]*np.exp(-prod)
        
        #update weights
        new_weights=[]
        for j in range(len(y)):
            if(y_pred2[j]==y[j]):
                new_weights.append(float((weight[j]*np.exp(-alpha[i]))/z))
            else:
                new_weights.append(float((weight[j]*np.exp(alpha[i]))/z))
        
        weight=new_weights
            
    return h_ens
    
    
    
    
    
def bagging(x, y, max_depth, num_trees):
    indices=[]
    dtree=[]
    w=[]
    for i in range(len(x)):
        indices.append(i) 
        w.append(1)
    for i in range(num_trees):
        sampled_list=np.random.choice(indices,len(x),replace=True)
        Xtrn2=[]
        ytrn2=[]
        for indice in sampled_list:
            Xtrn2.append(x[indice])
            ytrn2.append(y[indice])
        Xtrn2=np.asarray(Xtrn2)
        ytrn2=np.asarray(ytrn2)
        dtree.append(id3(Xtrn2,ytrn2,w=w))
    return dtree

def confusion_report(ytst2,y_pred3):
    cm=confusion_matrix(ytst2,np.asarray(y_pred3))
    print(cm)
    plt.matshow(cm)
    plt.title('Confusion matrix: Max_Depth=5, Bag size=5')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()  

def sklearn_confusion_bagging(Xtrn2,ytrn2,Xtst2,ytst2,max_depth,num_bags):

    clf = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth)
    bagC=BaggingClassifier(base_estimator=clf, n_estimators=num_bags, max_samples=len(Xtrn2), max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
    
    bagC.fit(Xtrn2,ytrn2)
    cm=confusion_matrix(ytst2,bagC.predict(Xtst2))
    print(cm)
    plt.matshow(cm)
    plt.title('Sklearn Bagging Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def sklearn_confusion_adaboost(Xtrn2,ytrn2,Xtst2,ytst2,max_depth,num_stumps):
   
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth)
    adaClf=AdaBoostClassifier(base_estimator=clf, n_estimators=num_stumps)
    
    adaClf.fit(Xtrn2,ytrn2)
    cm=confusion_matrix(ytst2,adaClf.predict(Xtst2))
    print(cm)
    plt.matshow(cm)
    plt.title('Sklearn Boosting Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + '/Users/rinkleseth/anaconda3/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid





if __name__ == '__main__':
    # Load the training data
    M2 = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = M2[:, 0]
    Xtrn2 = M2[:, 1:]

    M2 = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = M2[:, 0]
    Xtst2 = M2[:, 1:]
    
    #boosting-----------------------------------------------------------------
  
    h_ens=boosting(Xtrn2,ytrn2,2,10)
    y_pred2 = [predict_example(x,h_ens) for x in Xtst2]
    tst_err2=compute_error(ytst2,y_pred2)
    #print('Test Error for Boosting = {0:4.2f}%.'.format(tst_err2 * 100))
    
    #confusion_report(ytst2,y_pred2)
    #sklearn_confusion_adaboost(Xtrn2,ytrn2,Xtst2,ytst2,1,5)
    #----------------------------------------------------------------------------bagging
    y_pred3=[]
    dtree=bagging(Xtrn2,ytrn2,5,10)
    for x in Xtst2:
        y_pred2=[]
        for decision_tree in dtree:
            y_pred2.append(predict_example1(x,decision_tree)) 
        counts = np.bincount(y_pred2)
        y_pred3.append(np.argmax(counts))
    
   
    tst_err2=compute_error(ytst2,y_pred3)
    #confusion_report(ytst2,y_pred3)
    #sklearn_confusion_bagging(Xtrn2,ytrn2,Xtst2,ytst2,3,5)
    #print('Test Error for Bagging = {0:4.2f}%.'.format(tst_err2 * 100))
 
