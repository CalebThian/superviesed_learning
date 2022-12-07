#!/usr/bin/env python
# coding: utf-8
from seaborn import heatmap
from matplotlib.pyplot import show,xlabel,ylabel
from sklearn.model_selection import cross_val_score
from math import log
from random import randint,choices
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

data_path = "..//data//train.csv"

def str2num(data):
    data[1] = float(data[1])
    data[2] = float(data[2])
    data[3] = float(data[3])
    data[5] = int(data[5])
    data[6] = int(data[6])
    data[13] = int(data[13]) 
    data[20] = int(data[20])
    data[21] = int(data[21])
    data[23] = int(data[23])
    data[25] = float(data[25])
    data[26] = int(data[26])
    data[27] = int(data[27])
    data[28] = int(data[28])
    data[29] = int(data[29])
    data[42] = int(data[42])
    data[43] = int(data[43])

def label_encoder(results):
    # categorical column [4,7,8,9,10,11,12,14,15,16,17,18,19,22,24,30,31,32,33,34,35,36,37,38,39,40,41]
    cat = [4,7,8,9,10,11,12,14,15,16,17,18,19,22,24,30,31,32,33,34,35,36,37,38,39,40,41]
    Dict=dict()
        
    for col in cat:
        Dict[col] = dict()
        counter = 0          
        for r in results:
            if r[col] not in Dict[col].keys():
                Dict[col][r[col]]=counter
                counter += 1
            r[col] = Dict[col][r[col]]
    return Dict

def read_csv(path):
    with open(path , 'r') as f:
        header = []
        results = []
        for line in f:
            if header == []:
                header = line.split(',')
                header[-1] = header[-1][:-1]
            else:
                words = line.split(',')
                words[-1] = words[-1][:-1]
                str2num(words)
                results.append(words)
        label_encoder(results)
        return header,results

def train_test_split(data,val_size = 0.1, test_size = 0.1):
    train = int(len(data)*(1-val_size-test_size))
    val = int(len(data)*(1-test_size))
    
    data = shuffle(data)
    train_set = data[:train]
    val_set = data[train:val]
    test_set = data[val:]
    return train_set,val_set,test_set

def shuffle(data):
    length = len(data)
    for i in range(int(length*1.5)):
        ind1 = randint(0,length-1)
        ind2 = randint(0,length-1)
        temp = data[ind1]
        data[ind1] = data[ind2]
        data[ind2] = temp
    return data

def data2XY(data):
    X = []
    y = []
    for d in data:
        X.append(d[1:-1]) # ID is excluded
        y.append(d[-1])
    return X,y

def visualize_score(conf_mat):
    conf_mat_flatten = [conf_mat[0][0],conf_mat[0][1],conf_mat[1][0],conf_mat[1][1]]
    names = ['TN','FP','FN','TP']
    counts = ["{0:0.0f}".format(value) for value in conf_mat_flatten]
    total = conf_mat[0][0]+conf_mat[0][1]+conf_mat[1][0]+conf_mat[1][1]
    percentages = ["{0:.2%}".format(value/total) for value in conf_mat_flatten]
    labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(names,counts,percentages)]
    labels = [[labels[0],labels[1]],[labels[2],labels[3]]]
    heatmap(conf_mat, annot=labels, fmt='')
    xlabel('Predict', fontsize = 15) # x-axis label with fontsize 15
    ylabel('Truth', fontsize = 15) # y-axis label with fontsize 15

def data_process(val_size=0.1,test_size = 0.1,path=  "..//data//train.csv"):
    header,data = read_csv(path)
    train,test,_ = train_test_split(data,val_size=0.2,test_size=0)
    X_train,y_train = data2XY(train)
    X_test,y_test = data2XY(test)
    return X_train,y_train,X_test,y_test

def confusion_matrix(y_true,y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i,j in zip(y_true,y_pred):
        if i==j:
            if i == 1:
                TP += 1
            else:
                TN += 1
        else:
            if j == 1:
                FP += 1
            else:
                FN += 1
    if TP==0:
        recall = 0
        precision = 0
        f1_score = 0
    else:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1_score = 2*recall*precision/(recall+precision)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print("Recall = {0:.2%}".format(recall))
    print("Precision = {0:.2%}".format(precision))
    print("Accuracy = {0:.2%}".format(accuracy))
    print("F1-score = {0:.2%}".format(f1_score))
    return [[TN,FP],[FN,TP]]

def classify(clf,name,data_path =  ".//train.csv"):
    print(f"{name}:")
    X_train,y_train,X_test,y_test = data_process(val_size = 0.1,test_size = 0,path = data_path)
    clf.fit(X_train, y_train)
    
    '''
    # Test on train set
    y_pred = clf.predict(X_train)
    conf_mat = confusion_matrix(y_train,y_pred)
    visualize_score(conf_mat)
    show()
    '''
    
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test,y_pred)
    visualize_score(conf_mat)
    
def classify_k_fold(clf,name,k,data_path = './/train.csv'):
    print(f"{name}: F1-score")
    X,y,_,_ = data_process(val_size = 0,test_size = 0,path = data_path)
    for cv in k:
        mean_score = cross_val_score(clf, X, y, scoring="f1", cv = cv).mean()
        print("->{0}-fold cross validation:{1:.2%}".format(cv,mean_score))
        
#!/usr/bin/env python
# coding: utf-8
class NaiveBayesClassifier:
    def __init__(self):
        self.cat = [3,6,7,8,9,10,11,13,14,15,16,17,18,21,23,29,30,31,32,33,34,35,36,37,38,39,40]
        self.num = []
        self.e = 2.71828182846
        self.pi = 3.14159265359
        for i in range(42): #ID and is_claim is excluded
            if i not in self.cat:
                self.num.append(i)
        
    def fit(self,X,y):
        self.data = X
        self.y = y
        self.posterior = dict()
        self.claim = [0,0] #Calculation of is_claim in categorical fit
        for i,d in enumerate(self.data):
            claim = self.y[i]
            self.claim[claim] += 1 
        self.categorical_prob()
        self.numerical_prob()
        
    
    def categorical_prob(self):
        # Initialize
        for c in self.cat:
            self.posterior[c] = dict()
        
        for i,d in enumerate(self.data):
            claim = self.y[i]
            for c in self.cat:
                if d[c] not in self.posterior[c].keys():
                    self.posterior[c][d[c]]=[0,0]
                self.posterior[c][d[c]][claim] += 1
        
        # Assume all features are independent
        # P(x_i|C_j) = count(x_i & C_j)/count(C_j)
        for k,v in self.posterior.items():
            for x,count in v.items():
                count[0] = count[0]/self.claim[0]
                count[1] = count[1]/self.claim[1]
            #print(k,v)
        
    def numerical_prob(self):
        # Assume all the continuous features are gaussian distribution
        # take all the data into a list
        subdata = dict()
        for c in self.num:
            # The first subarray store the data of is_claim=0, the second store the data of is_claim=1
            subdata[c] = [[0],[0]] # The first element of each subarray is the sum of the rest elements

        for i,d in enumerate(self.data):
            for c in self.num:
                claim = self.y[i]
                subdata[c][claim][0] += d[c]
                subdata[c][claim].append(d[c])

        # Calculate mean and variance
        for c in self.num:
            self.posterior[c] = [[0,1],[0,1]] # mean and variance of is_claim = 0, mean and variance of is_claim = 1
            for i in range(2):
                mean = subdata[c][i][0]/(len(subdata[c][i])-1)
                variance = 0
                for j in range(1,len(subdata[c][i])):
                    variance += ((subdata[c][i][j]-mean)**2)
                variance /= (len(subdata[c][i])-1)
                self.posterior[c][i] = [mean,variance]
                
    
    def predict_one(self,data):
        p0 = 1
        p1 = 1
        for c in self.cat:
            p0 *= self.posterior[c][data[c]][0]
            p1 *= self.posterior[c][data[c]][1]
        
        for n in self.num:
            mean0 = self.posterior[c][0][0]
            var0 = self.posterior[c][0][1]
            mean1 = self.posterior[c][1][0]
            var1 = self.posterior[c][1][1]
            p0 *= (self.e**(-((data[c]-mean0)**2)/2/var0)/(2*self.pi*var0)**0.5)
            p1 *= (self.e**(-((data[c]-mean1)**2)/2/var1)/(2*self.pi*var1)**0.5)
            #if p0>p1:
            #    cmp = '>'
            #else:
            #    cmp = '<'
            #print(f"{p0}{cmp}{p1}")
        
        p0 *= self.claim[0]/(self.claim[0]+self.claim[1])
        p1 *= self.claim[1]/(self.claim[0]+self.claim[1])
        if p0>p1:
            return 0
        else:
            return 1
        
    def score(self,X,y_true):
        y_pred = []

        for x in X:
            y_pred.append(self.predict_one(d))
        
        return confusion_matrix(y_true,y_pred)
    
    def predict(self,data):
        y_pred = []
        for d in data:
            y_pred.append(self.predict_one(d))
        return y_pred
    
    def get_params(self,deep=False):
        return dict()

class myRandomForestClassifier:
    def __init__(self):
        self.n_estimators = 10
        self.max_features = int(42**0.5)
        self.max_depth = 10
        self.min_samples_split = 2
        self.num_features = 42
        self.feature_cat = [3,6,7,8,9,10,11,13,14,15,16,17,18,21,23,29,30,31,32,33,34,35,36,37,38,39,40]
        self.feature_num = []
        for i in range(42): #ID and is_claim is excluded
            if i not in self.feature_cat:
                self.feature_num.append(i)
                
        self.sample_list = []
        for i in range(1,self.num_features):
            self.sample_list.append(i)

    def fit(self,X,y):
        self.seq = []
        for i in range(len(X)):
            self.seq.append(i)
        self.tree_ls = self.random_forest(X, y, self.n_estimators, self.max_depth, self.min_samples_split, self.max_features)

    def predict(self,X):
        return self.predict_rf(self.tree_ls,X)

    def log2(self,p):
        return log(p)/log(2)

    def entropy(self,p):
        if p == 0:
            return 0
        elif p == 1:
            return 0
        else:
            return - (p * self.log2(p) + (1 - p) * self.log2(1-p))

    def information_gain(self,left_child, right_child):
        parent = left_child + right_child
        p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
        p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
        p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
        IG_p = self.entropy(p_parent)
        IG_l = self.entropy(p_left)
        IG_r = self.entropy(p_right)
        return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r

    def slices(self,data,indices):
        d = []
        for ind in indices:
            d.append(data[ind])
        return d

    def sampling(self,size):
        length = len(self.sample_list)
        for i in range(length):
            ind1 = randint(0,length-1)
            ind2 = randint(0,length-1)
            temp = self.sample_list[ind1]
            self.sample_list[ind1] = self.sample_list[ind2]
            self.sample_list[ind2] = temp
        ind_start = randint(0,length-7)
        return self.sample_list[ind_start:ind_start+6]

    def data_col(self,data,feature_ind):
        col_data = []
        ls = set()
        for d in data:
            ls.add(d[feature_ind])
            col_data.append(d[feature_ind])

        return col_data,ls

    def draw_bootstrap(self,X_train, y_train):
        bootstrap_indices = choices(self.seq,k=len(X_train))
        oob_indices = list(set(self.seq)-set(bootstrap_indices))
        X_bootstrap = self.slices(X_train,bootstrap_indices)
        y_bootstrap = self.slices(y_train,bootstrap_indices)
        X_oob = self.slices(X_train,oob_indices)
        y_oob = self.slices(y_train,oob_indices)
        return X_bootstrap, y_bootstrap, X_oob, y_oob

    def oob_score(self,tree, X_test, y_test):
        mis_label = 0
        for i in range(len(X_test)):
            pred = self.predict_tree(tree, X_test[i])
            if pred != y_test[i]:
                mis_label += 1
        return mis_label / len(X_test)

    def find_split_point(self,X_bootstrap, y_bootstrap, max_features):
        feature_ls = self.sampling(max_features)

        best_info_gain = -999
        node = None
        for feature_ind in feature_ls:
            col_data,unique_col_data = self.data_col(X_bootstrap,feature_ind)
            for split_point in unique_col_data:
                left_child = {'X_bootstrap': [], 'y_bootstrap': []}
                right_child = {'X_bootstrap': [], 'y_bootstrap': []}

                # split children for continuous variables
                if feature_ind in self.feature_num:
                    for i, value in enumerate(col_data):
                        if value <= split_point:
                            left_child['X_bootstrap'].append(X_bootstrap[i])
                            left_child['y_bootstrap'].append(y_bootstrap[i])
                        else:
                            right_child['X_bootstrap'].append(X_bootstrap[i])
                            right_child['y_bootstrap'].append(y_bootstrap[i])

                # split children for categorical variables
                else:
                    for i, value in enumerate(col_data):
                        if value == split_point:
                            left_child['X_bootstrap'].append(X_bootstrap[i])
                            left_child['y_bootstrap'].append(y_bootstrap[i])
                        else:
                            right_child['X_bootstrap'].append(X_bootstrap[i])
                            right_child['y_bootstrap'].append(y_bootstrap[i])

                split_info_gain = self.information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
                if split_info_gain > best_info_gain:
                    best_info_gain = split_info_gain
                    node = {'information_gain': split_info_gain,
                            'left_child': left_child,
                            'right_child': right_child,
                            'split_point': split_point,
                            'feature_ind': feature_ind}

        return node

    def terminal_node(self,node):
        y_bootstrap = node['y_bootstrap']
        pred = max(y_bootstrap, key = y_bootstrap.count)
        return pred


    def split_node(self,node, max_features, min_samples_split, max_depth, depth):
        left_child = node['left_child']
        right_child = node['right_child']

        del(node['left_child'])
        del(node['right_child'])

        if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
            empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
            node['left_split'] = self.terminal_node(empty_child)
            node['right_split'] = self.terminal_node(empty_child)
            return

        if depth >= max_depth:
            node['left_split'] = self.terminal_node(left_child)
            node['right_split'] = self.terminal_node(right_child)
            return node

        if len(left_child['X_bootstrap']) <= min_samples_split:
            node['left_split'] = node['right_split'] = self.terminal_node(left_child)
        else:
            node['left_split'] = self.find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
            self.split_node(node['left_split'], max_features, min_samples_split, max_depth, depth + 1)

        if len(right_child['X_bootstrap']) <= min_samples_split:
            node['right_split'] = node['left_split'] = self.terminal_node(right_child)
        else:
            node['right_split'] = self.find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
            self.split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

    def build_tree(self,X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
        root_node = self.find_split_point(X_bootstrap, y_bootstrap, max_features)
        self.split_node(root_node, max_features, min_samples_split, max_depth, 1)
        return root_node

    def random_forest(self,X_train, y_train, n_estimators, max_depth, min_samples_split, max_features):
        tree_ls = list()
        oob_ls = list()
        for i in range(n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.draw_bootstrap(X_train, y_train)
            tree = self.build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split,max_features)
            tree_ls.append(tree)
            oob_error = self.oob_score(tree, X_oob, y_oob)
            oob_ls.append(oob_error)
        #print("OOB estimate: {:.2f}".format(sum(oob_ls)/len(oob_ls)))# mean
        return tree_ls

    def predict_tree(self,tree, X_test):
        feature_ind = tree['feature_ind']

        if X_test[feature_ind] <= tree['split_point']:
            if type(tree['left_split']) == dict:
                return self.predict_tree(tree['left_split'], X_test)
            else:
                value = tree['left_split']
                return value
        else:
            if type(tree['right_split']) == dict:
                return self.predict_tree(tree['right_split'], X_test)
            else:
                return tree['right_split']

    def predict_rf(self,tree_ls, X_test):
        pred_ls = []
        for i in range(len(X_test)):
            ensemble_preds = [self.predict_tree(tree, X_test[i]) for tree in tree_ls]
            final_pred = max(ensemble_preds, key = ensemble_preds.count)
            pred_ls.append(final_pred)
        return pred_ls
    
    def get_params(self,deep=False):
        return dict()
    
k = [3,5,10]


# In[3]:


clf = NaiveBayesClassifier()
name = "Naive Bayes Clasifier"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[4]:


clf = myRandomForestClassifier()
name = "my Random Forest Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)

# In[6]:


clf = RandomForestClassifier(n_estimators = 10,max_depth=10, random_state=0)
name = "Random Forest Classifier of scikit-learn"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[7]:


clf = XGBClassifier()
name = "XGBoost Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[8]:


clf = CatBoostClassifier(verbose=0)
name = "Catboost Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[9]:


clf = lgb.LGBMClassifier(is_unbalance=True)
name = "LightGBM Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)

