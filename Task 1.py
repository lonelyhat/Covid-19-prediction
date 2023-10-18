import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

#reading from the file
df=pd.read_csv('AAA_owid-covid-data.csv',dtype='unicode')
columns_from_main=['location','date','total_cases','new_cases','new_deaths','new_vaccinations']


#filtering the data with the following filters:
#1)Last 180 days for each of the 180 countries
#2)removing all the rows with NaN values
#numbering the dates
df['date']=pd.to_datetime(df['date'])#this and the next line extracts the values that are between a specified time range
mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')
df_location=df.loc[(mask)&((df['location'] == 'Singapore')|(df['location'] =='South Korea')|(df['location'] =='India')|(df['location'] =='Philippines')|(df['location'] =='Japan')|(df['location'] =='Vietnam')|(df['location']=='United Kingdom')|(df['location']=='Pakistan')|(df['location']=='Germany')|(df['location']=='Nigeria')|(df['location']=='Italy')|(df['location']=='China')|(df['location']=='Indonesia')|(df['location']=='Thailand')|(df['location']=='United Arab Emirates')|(df['location']=='Portugal')|(df['location']=='Cuba')|(df['location']=='Ethiopia')),columns_from_main]
df_location.dropna(subset=['new_vaccinations'],inplace=True)
#this removes the rows that have NaN values
print(df_location.size)
print("The locations under observation are:\n",df_location)#only to see the data set



#This part is from the cohort problem set:
def get_features_targets(df, feature_names, target_names):
    df_feature=df[feature_names]
    df_target=df[target_names]
    return df_feature, df_target

def normalize_z(df):
    return (df-df.mean())/df.std()

def prepare_feature(df_feature):
    df_feature=df_feature.to_numpy()
    c_ones=np.ones((df_feature.shape[0],1))
    df_feature=np.hstack((c_ones,df_feature))
    return df_feature

def prepare_target(df_target):
    df_target=df_target.to_numpy()
    return df_target

def predict(df_feature, beta):
    X=prepare_feature(normalize_z(df_feature))
    return predict_norm(X,beta)

def predict_norm(X, beta):
    return np.matmul(X,beta)

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    indexes=df_feature.index
    if random_state!=None:
        np.random.seed(random_state)
    num_rows = len(indexes)
    k = int(test_size * num_rows)
    test_indices = np.random.choice(indexes, k, replace = False)
    train_indices = set(indexes) - set(test_indices)
    
    df_feature_train = df_feature.loc[train_indices, :]
    df_feature_test = df_feature.loc[test_indices, :]
    df_target_train = df_target.loc[train_indices, :]
    df_target_test = df_target.loc[test_indices, :]
    return df_feature_train, df_feature_test, df_target_train, df_target_test
  
def r2_score(y, ypred):
    SS_res=np.sum((y-ypred)**2)
    SS_tot=np.sum((y-np.mean(y))**2)
    return (1-(SS_res/SS_tot))

def mean_squared_error(target, pred):
    return ((np.sum(target-pred)**2)/(target.shape[0]))

def gradient_descent(X, y, beta, alpha, num_iters):
    n=X.shape[0]
    J_storage=np.zeros((num_iters,1))
    for i in range(num_iters):
        derivative=np.matmul(X.T,(np.matmul(X,beta)-y))
        beta=beta-(alpha/n*derivative)
        J_storage[i]=compute_cost(X,y,beta)
    return beta, J_storage

def compute_cost(X, y, beta):
    J = 0
    m = X.shape[0]
    error = np.matmul(X,beta) - y
    error_sq = np.matmul(error.T, error)
    plt.scatter(X[:,1], y)
    J = 1/(2*m) * error_sq
    J = J[0][0]
    return J




#this part is to extract the data from df_location that is relevant to the multiple linear regression model
#convert the string value to a number:
#
df_location["new_deaths"]=pd.to_numeric(df_location["new_deaths"], downcast='integer')
df_location["new_cases"]=pd.to_numeric(df_location["new_cases"], downcast='integer')
df_location['new_vaccinations']=pd.to_numeric(df_location['new_vaccinations'], downcast='integer')
#
features=['new_cases','new_vaccinations']
target=['new_deaths']
df_features,df_target=get_features_targets(df_location,features,target)
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features,df_target,random_state=100,test_size=0.3)
df_features_train_z=normalize_z(df_features_train)
X=prepare_feature(df_features_train_z)
target=prepare_target(df_target_train)

iterations=1500
alpha=0.01
beta=np.zeros((3,1))

beta,J_storage=gradient_descent(X,target,beta,alpha,iterations)
pred=predict(df_features_train_z,beta)
print (pred)

#here come all the graphical representations:






'''df_model.to_csv('normalized data.csv',index=False)



#building the matrix
X=df_model.iloc[:,0:2]
ones=np.ones([X.shape[0],1])
X=np.concatenate((ones,X),axis=1)

y=df_model.iloc[:,2:3].values
theta=np.zeros([1,3])

alpha=0.1
iters=1000

def compute_cost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = compute_cost(X, y, theta)
    return theta,cost

g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = compute_cost(X,y,g)
print(finalCost)

fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 
plt.show()'''

'''#This part is for plotting the lineplot of 
sns.set()
plot=sns.lineplot(y='date',x='total_vaccinations',hue='location',data=df_location)#plot to show new cases vs new deaths
#this clears up the number of x labels and y labels
#
#
for ind,label in enumerate(plot.get_xticklabels()):
    if ind%200!=0:
        label.set_visible(False)

for ind,label in enumerate(plot.get_yticklabels()):
    if ind%10!=0:
        label.set_visible(False)
#
#
plt.show()
print(df_location['total_vaccinations'])
'''
