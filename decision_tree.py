# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import operator
import matplotlib.pyplot as plt

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
    
    #print(x)
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    dict_y=partition(y)
    #print(len(dict_y[0]))
    #print(len(dict_y))
    prob_sum=0
    total_val=sum([len(x) for x in dict_y.values()])
    for key in dict_y:
        count_key=len(dict_y[key])
        prob=-1*float(count_key/total_val)*np.log2(float(count_key/total_val))
        prob_sum+=prob
    return(prob_sum)
    # INSERT YOUR CODE HERE
    #raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    dict_x=partition(x)
    #print(dict_x)
    entropy_y=entropy(y)
    #print("y",entropy_y)
    ans=0
    #total_val=sum([len(x) for x in dict_x.values()])
    for key in dict_x:
        y_ans=[]
        for i in dict_x[key]:
            y_ans.append(y[i])
        
        #print(y_ans)
        entropy_x=entropy(y_ans)
        #print(key)
        #print("entropy_x",entropy_x)
        prob=len(dict_x[key])/len(x)
        #print(prob)
        ans+=prob*entropy_x
    mi=entropy_y-ans
    return mi
    # INSERT YOUR CODE HERE
    #raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=3):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b), 
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    dtree=dict()
    #print("depth:",depth)
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
            #print(attribute_value_pairs) #list
            
    mi_dict=dict()
   
    for i,value in attribute_value_pairs:
        #print(i,value)
        col=x[:,i] #each attribute column
        #print(col)
        bin_col=np.zeros((len(col),), dtype=int)
        #print(bin_col)
        for j in range(len(col)):
            if col[j]==value:
                bin_col[j]=1
        #print(bin_col)  
        mi=mutual_information(bin_col,y)
        mi_dict[(i,value)]=mi
    #print(mi_dict)
    #----------------------------------------------------------
    
    #attribute_value_pairs=best_attr_value(x)
    #print(attribute_value_pairs)
    attr,value=max(mi_dict.items(), key=operator.itemgetter(1))[0]
    #print("best attribute value:",attr,value)
    best_col=x[:,attr]
    #print(best_col)
    
    #-----------------------------------split on best attribute
    xsubset_true=[]
    xsubset_false=[]
    ysubset_true=[]
    ysubset_false=[]
    #dict_x[value]
    for i in range(len(x)):
        if best_col[i]==value:
            xsubset_true.append(x[i])
            ysubset_true.append(y[i])
        else:
            xsubset_false.append(x[i])
            ysubset_false.append(y[i])
            
    #print("Depth",depth,ysubset_true)
    
    
    #print
    xsubset_true=np.asarray(xsubset_true)
    xsubset_false=np.asarray(xsubset_false)
    ysubset_true=np.asarray(ysubset_true)
    ysubset_false=np.asarray(ysubset_false)
    #print(len(xsubset_false))
    #print(len(ysubset_false))
    attribute_value_pairs_child=[]
    attribute_value_pairs_child=attribute_value_pairs.copy()
    attribute_value_pairs_child.remove((attr,value))
    
    #----------------------------------------recursive calls for subtree
    dtree.setdefault((attr,value,True),{})
    dtree.setdefault((attr,value,False),{})
    
    dtree[(attr,value,True)]=id3(xsubset_true,ysubset_true,attribute_value_pairs_child,depth+1,max_depth=max_depth)
    dtree[(attr,value,False)]=id3(xsubset_false,ysubset_false,attribute_value_pairs_child,depth+1,max_depth=max_depth)
    
    #attribute_value_pairs.remove((attr,value))
 
    return dtree
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree

    """
    key=list(tree.keys())[0]
    (col,attr,val)=key

    if x[col]==attr:
        #print("coll=attr")
        if type(tree[(col,attr,True)]) is dict:
            return predict_example(x,tree[(col,attr,True)])

        else:

            return tree[(col,attr,True)]

    else:
        #print("coll!=attr")
        if type(tree[(col,attr,False)]) is dict:
            return predict_example(x,tree[(col,attr,False)])

        else:
            return tree[(col,attr,False)]
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    #print(y_true)
    #print(y_pred)
    n=len(y_pred)
    count=0
    for i in range(n):
        if y_true[i]!=y_pred[i]:
            count+=1
    return float(count)/float(n)


    # INSERT YOUR CODE HERE
    #raise Exception('Function not yet implemented!')


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


def plot2(Xtrn,ytrn,Xtst,ytst,title):
    trainErr = {}   # Train error of the models
    testErr = {}         # Test error of all the models
    for d in range(1,11):  
        trn_d

        tree = id3(Xtrn,ytrn,max_depth=d)
        y_pred1 = [predict_example(x, trn_dtree) for x in Xtrn]
        trainErr[d] = compute_error(ytrn, y_pred1)
        
        y_pred2 = [predict_example(x, trn_dtree) for x in Xtst]
        testErr[d] = compute_error(ytst, y_pred2)
   
    plt.figure()
    plt.plot(trainErr.keys(), trainErr.values(), marker='o', linewidth=3, markersize=12)
    plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
    plt.xlabel('Depth', fontsize=16)
    plt.title(title)
    plt.ylabel('Validation/Test error', fontsize=16)
    plt.xticks(list(trainErr.keys()), fontsize=12)
    plt.legend(['Training Error', 'Test Error'], fontsize=16)



if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    #M2 = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #ytrn2 = M2[:, 0]
    #Xtrn2 = M2[:, 1:]
    
        # Load the test data
    #M2 = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #ytst2 = M2[:, 0]
    #Xtst2 = M2[:, 1:]
    
    
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    #decision_tree2 = id3(Xtrn2, ytrn2, max_depth=3)
    # Pretty print it to console
    pretty_print(decision_tree)
    

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    #dot_str = to_graphviz(decision_tree2)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    #y_pred2= [predict_example(x2, decision_tree2) for x2 in Xtst2]
    #tst_err2 = compute_error(ytst2, y_pred2)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
   # print('Test Error = {0:4.2f}%.'.format(tst_err2 * 100))
