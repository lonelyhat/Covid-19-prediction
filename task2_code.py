import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import math

df=pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',dtype='unicode')
columns_from_main=['location','date','new_cases','total_vaccinations',"population"]

#Filter data
df['date']=pd.to_datetime(df['date'])
mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')
df_location=df.loc[(mask)&(df['location']=='Japan'),columns_from_main]
df_location.dropna(subset=['total_vaccinations'],inplace=True)
df_location.dropna(subset=['population'],inplace=True)
df_location.dropna(subset=['new_cases'],inplace=True)
#convert the string value to a number:
df_location["new_cases"]=pd.to_numeric(df_location["new_cases"], downcast='integer')
df_location['total_vaccinations']=pd.to_numeric(df_location['total_vaccinations'], downcast='integer')
df_location['population']=pd.to_numeric(df_location['population'], downcast='integer')
print(df_location.dtypes)
mask=(df_location["new_cases"] > 0)&(df_location['total_vaccinations'] > 0)
df_location=df_location.loc[mask,columns_from_main]

print(df_location.size)
print("The locations under observation are:\n",df_location)

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
    actual_mean = np.mean(y)
    # since y, ypred are both nparray, y-ypred does element wise SUB
    ssres = np.sum((y-ypred)**2)
    sstot = np.sum((y-actual_mean)**2)
    return 1 - (ssres/sstot)

def mean_squared_error(target, pred):
    num_of_samples = target.shape[0] # number of samples == number of rows in target_y
    return (1/num_of_samples) * np.sum((target-pred)**2)

def gradient_descent(X, y, beta, alpha, num_iters):
    # For linreg, X is a n by 2 matrix, beta is a 2 by 1 vector, y (actual_target) n by 1 vector, alpha is Float, num_iters is an Int
    # beta -> initial guess of beta 
    ### BEGIN SOLUTION
    number_of_samples = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    # iterate the grad desc until num_iters
    # or, until convergence (other case)
    for i in range(num_iters):
        # this derivate is derived from the squared error function 
        # STEP 2
        # Y_pred = X x Beta
        # diff Y_pred/Beta --> X.T x (X x Beta) 
        # transpose X and put on the left hand side of matrix mul
        derivative_cost_wrt_Beta = (1/number_of_samples) * np.matmul(X.T, (np.matmul(X, beta) - y))
        # update beta
        # STEP 3
        beta = beta - alpha * derivative_cost_wrt_Beta
        J_storage[i] = compute_cost(X, y, beta)
    ### END SOLUTION
    return beta, J_storage

def compute_cost(X, y, beta):
    # for LinReg: X is n by 2, y is a vector of n elements, beta is 2 by 1
    J = 0
    ### BEGIN SOLUTION
    number_of_samples = X.shape[0]
    # Y_pred - Y_actual
    # Y_pred = Xb
    Y_pred = np.matmul(X, beta)
    diff_between_pred_actual_y = Y_pred - y
    diff_between_pred_actual_y_sq = np.matmul(diff_between_pred_actual_y.T, diff_between_pred_actual_y)
    J = (1/(2*number_of_samples)) * diff_between_pred_actual_y_sq
    ### END SOLUTION
    # J is an error, it is a scalar, so extract the only element of J that was a numpy array
    return J[0][0]

def logarithm(df_target, target_name):
    ### BEGIN SOLUTION
    df_out = df_target.copy()
    df_out.loc[:, target_name] = df_target[target_name].apply(lambda x: math.log(x))
    ### END SOLUTION
    return df_out

def predict_num_new_cases(vac_num,population,country):
    if country == "Japan":
        #For Japan
        return int(math.exp(-1.64771199*(int(vac_num)/int(population)-1.225269)/0.216455+7.74928016))
    elif country == "India":
        #For India
        return int(math.exp(-0.78281369*(int(vac_num)/int(population)-0.404356)/0.205019+10.54177744))
    else:
        print("Please input the correct name of country.")
    
    

#Trial
df_location.loc[:,"total_vaccinations/population"] = df_location['total_vaccinations'].div(df_location['population'])
#print(df_location)
features=['total_vaccinations/population']
target=['new_cases']
df_features,df_target=get_features_targets(df_location,features,target)
#myplot = sns.scatterplot(x="total_vaccinations/population", y="new_cases", data=df_location)

df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features,df_target,random_state=100,test_size=0.3)
#Normalize both train and test features
df_features_train = normalize_z(df_features_train)
df_features_test = normalize_z(df_features_test)

X=prepare_feature(df_features_train)
target=prepare_target(df_target_train)

iterations=1500
alpha=0.01
beta=np.zeros((2,1))

beta,J_storage=gradient_descent(X,target,beta,alpha,iterations)
pred=predict(df_features_test,beta)
print("Beta0 and beta1 is equal to: ")
print(beta)
plt.scatter(df_features_test, df_target_test)
plt.plot(df_features_test, pred, color="orange")

the_target = prepare_target(df_target_test)
r2= r2_score(the_target, pred)
print("R2 Coefficient of Determination:")
print(r2)
mse = mean_squared_error(the_target, pred)
print("Mean Squared Error:")
print(mse)

the_target = prepare_target(df_target_test)
r2= r2_score(the_target, pred)
print(r2)
mse = mean_squared_error(the_target, pred)
print(mse)


#Improvement
df_location.loc[:,"total_vaccinations/population"] = df_location['total_vaccinations'].div(df_location['population'])
features=['total_vaccinations/population']
target=['new_cases']
df_features,df_target=get_features_targets(df_location,features,target)
# Try to set the range for x
#print(df_location)
if df_location['location'].all()=='Japan':
    df_features_improved = df_features.loc[df_features["total_vaccinations/population"]>0.8]
    print(df_location.loc[df_location["total_vaccinations"]/df_location["population"]>0.8])
else:
    df_features_improved = df_features
df_target_improved = df_target.loc[set(df_features_improved.index),:]
#print(df_features_improved)
#myplot = sns.scatterplot(x="total_vaccinations/population", y="new_cases", data=df_location)
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features_improved,df_target_improved,random_state=100,test_size=0.3)

#Apply logarithm to change y to lny
df_target_train = logarithm(df_target_train, "new_cases")
df_target_test = logarithm(df_target_test, "new_cases")
#print(df_target_train)
print(df_features_train.mean())
print(df_features_train.std())
#Normalize both train and test features
df_features_train = normalize_z(df_features_train)
df_features_test = normalize_z(df_features_test)


X=prepare_feature(df_features_test)
Y=prepare_target(df_target_test)
#print(df_target_test)
iterations=1500
alpha=0.01
beta=np.zeros((2,1))

beta,J_storage=gradient_descent(X,Y,beta,alpha,iterations)
pred=predict(df_features_test,beta)
print("Beta0 and beta1 is equal to: ")
print(beta)


plt.scatter(df_features_test, df_target_test)
plt.plot(df_features_test, pred, color="orange")
the_target = prepare_target(df_target_test)
r2= r2_score(the_target, pred)

print("R2 Coefficient of Determination:")
print(r2)
mse = mean_squared_error(the_target, pred)
print("Mean Squared Error:")
print(mse)

print("Predicted result: "+str(predict_num_new_cases(1088236037,1393409033)))






