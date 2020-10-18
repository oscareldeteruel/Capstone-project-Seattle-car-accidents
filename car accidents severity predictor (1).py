#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries for url import and dataframes management.
import numpy as pd
import pandas as pd


# In[2]:


#Read dataset from url
url="https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv"
c=pd.read_csv(url)


# In[3]:


#Explore data content
c.head()


# In[4]:


#Visualize number of records and columns
c.shape


# In[5]:


# Explore the type of records we have in each column of the dataframe
c.dtypes


# In[6]:


# Select from the total list of variables and their associated values, only those which will be applicable to our project analysis
car_accidents  = c[['SEVERITYCODE','ADDRTYPE','X','Y','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','JUNCTIONTYPE','WEATHER','ROADCOND','LIGHTCOND']]
car_accidents  .head()


# In[7]:


# Visualise the number of records we have  in each of those variables 
car_accidents.count().to_frame()


# In[8]:


#We can see that not all variables are filled in for all records properly, the ones where the number of records are below 194673 which is the total number of records have to be evaluated
#Now we start exploring the values in the variables below the total number of records
car_accidents['ADDRTYPE'].value_counts().to_frame()


# In[9]:


car_accidents['JUNCTIONTYPE'].value_counts().to_frame()


# In[10]:


car_accidents['WEATHER'].value_counts().to_frame()


# In[11]:


car_accidents['ROADCOND'].value_counts().to_frame()


# In[12]:


car_accidents['LIGHTCOND'].value_counts().to_frame()


# In[13]:


#Some of the variables contain values like unknown and other which won't add any value to the model and can lead to wrong predictions due to overfitting or noise
#Remove all records where the variable WEATHER is unknown or other
car_accidents= car_accidents[car_accidents.WEATHER != 'Unknown']

car_accidents= car_accidents[car_accidents.WEATHER != 'Other']
#let's check the shape after performing the removal
car_accidents.shape


# In[14]:


# we remove all unknown values in roadcon with unknown or other
car_accidents= car_accidents[car_accidents.ROADCOND != 'Unknown'] 
car_accidents= car_accidents[car_accidents.ROADCOND != 'Other'] 
car_accidents.shape


# In[15]:


# we remove all unknown values in light cond
car_accidents= car_accidents[car_accidents.LIGHTCOND != 'Unknown'] 
car_accidents= car_accidents[car_accidents.LIGHTCOND != 'Other'] 
car_accidents.shape


# In[16]:


# we remove all unknown values in ADDRTYPE
car_accidents= car_accidents[car_accidents.ADDRTYPE != 'Unknown'] 
car_accidents.shape


# In[17]:


car_accidents.count().to_frame()


# In[18]:


#drop all the rows where there is at least an  empty field in any of the columns of analysis
car_accidents = car_accidents.dropna(how='any', subset=['ADDRTYPE', 'X' , 'Y','JUNCTIONTYPE', 'WEATHER','ROADCOND', 'LIGHTCOND' ])
car_accidents.count().to_frame()


# In[19]:


# the value of the column junction type, doesn't add too much value further than the address type ( block or interseciton ) to the analysis, so we will drop it from the analysis later on. 


# In[20]:


# the number of total records after the cleaning is 164731. Now we need to check how balanced is our dataset in terms of the values for the variable to be predicted.
car_accidents['SEVERITYCODE'].value_counts().to_frame()


# In[21]:


#Now we need to balance the records since we have two times more records with severity code 1 than 2. 
#We will in this case Down-sample Majority Class 
#Downsample majority classPython

from sklearn.utils import resample

# Separate majority and minority classes
car_accidents_majority = car_accidents[car_accidents.SEVERITYCODE==1]
car_accidents_minority = car_accidents[car_accidents.SEVERITYCODE==2]
 
# Downsample majority class
car_accidents_majority_downsampled = resample(car_accidents_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=54544,     # to match minority class number of records
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
car_accidents_downsampled = pd.concat([car_accidents_majority_downsampled, car_accidents_minority])
 
# Display new class counts
car_accidents_downsampled.SEVERITYCODE.value_counts()


# In[22]:


# Now we have the same number of samples and the data is completely clean to be used.
# Exploratory analysis before going into the dataset


# In[23]:


#let's split the analysis for one side taking all numerical variables and we will check how the different values are distributed and how they are related to the study variable severity code different values
#We select the numberical values only
car_accidents_downsampled_num = car_accidents_downsampled.select_dtypes(include = ['float64', 'int64'])
car_accidents_downsampled_num.head()


# In[24]:


# we plot each of the variables versus the severity code values and we count the number or records 
car_accidents_downsampled_num.hist(figsize=(16, 20), bins=100, xlabelsize=8, ylabelsize=8)


# In[25]:


# As we can see the variables, offer almost not information about the target variable, most of the accidents occurred with 2 vehicles, no pedestrians and no bycicles and very little number of people.
# The location variables graph says that  a big chunk of the accidents occures close to the central locations where the density of cars is always higher.  
# Since this is not very revealing let's see the correlation versus the target variable. 
# we will use two methods pearson and kendall


# In[26]:


#Pearson correlation
car_accidents_downsampled_num.corr(method ='pearson') 


# In[27]:


#kendall correlation
car_accidents_downsampled_num.corr(method ='kendall') 


# In[28]:


# low correlations between  the numerical variables and the target variable, none of them shows a value above 0.5 which might start to be relevant. 
# Among the variables themselves the correlations are also low.
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[29]:


# correlation matrix in a heatmap
# as of 0.5 the correlation becomes relevant, since thereis no values above this threshold I downgrade the threshold to 0.2 to plot the map 
from matplotlib import pyplot as plt
corr = car_accidents_downsampled_num.drop('SEVERITYCODE', axis=1).corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.2) | (corr <= -0.2)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# In[30]:


# we see all have low correlationm, the highest is around 0.4... with number of people and vehicles involved. 


# In[31]:


#let's check now the non categorical variables do 


# In[32]:


import seaborn as sns
ax = sns.countplot(y="WEATHER", hue="SEVERITYCODE", data=car_accidents_downsampled)


# In[33]:


ax = sns.countplot(y="JUNCTIONTYPE", hue="SEVERITYCODE", data=car_accidents_downsampled)


# In[34]:


ax = sns.countplot(y="ADDRTYPE", hue="SEVERITYCODE", data=car_accidents_downsampled)


# In[35]:


ax = sns.countplot(y="ROADCOND", hue="SEVERITYCODE", data=car_accidents_downsampled)


# In[36]:


ax = sns.countplot(y="LIGHTCOND", hue="SEVERITYCODE", data=car_accidents_downsampled)


# In[37]:


# We plot in a desnsity map the location points to be able to observe its density 

with sns.axes_style('white'):
    sns.jointplot("X", "Y", car_accidents_downsampled, kind='kde');


# In[38]:


# Now we start to encode our categorical values  variables of study to be used in the model  
#"WEATHER","ROADCOND","LIGHTCOND","SEVERITYCODE",ADDRTYPE
# For traceability, we will prepare the dataset already encoded into a new dataframe.

car_accidents_downsampledtrain_test= car_accidents_downsampled


#Encoding Road Conditions 
car_accidents_downsampledtrain_test["ROADCOND"].replace("Dry", 0, inplace=True)
car_accidents_downsampledtrain_test["ROADCOND"].replace("Wet", 2, inplace=True)
car_accidents_downsampledtrain_test["ROADCOND"].replace("Ice", 2, inplace=True)
car_accidents_downsampledtrain_test["ROADCOND"].replace("Snow/Slush", 1, inplace=True)
car_accidents_downsampledtrain_test["ROADCOND"].replace("Standing Water", 2, inplace=True)
car_accidents_downsampledtrain_test["ROADCOND"].replace("Sand/Mud/Dirt", 1, inplace=True)
car_accidents_downsampledtrain_test["ROADCOND"].replace("Oil", 1, inplace=True)


#Encoding Weather 
car_accidents_downsampledtrain_test["WEATHER"].replace("Clear", 0, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Raining", 3, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Overcast", 1, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Snowing", 3, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Fog/Smog/Smoke", 2, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Sleet/Hail/Freezing Rain", 3, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Blowing Sand/Dirt", 2, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Severe Crosswind", 2, inplace=True)
car_accidents_downsampledtrain_test["WEATHER"].replace("Partly Cloudy", 2, inplace=True)

#Encoding Light Conditions
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Daylight", 0, inplace=True)
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Dark - Street Lights On", 1, inplace=True)
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Dark - No Street Lights", 2, inplace=True)
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Dusk", 1, inplace=True)
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Dawn", 1, inplace=True)
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Dark - Street Lights Off", 2, inplace=True)
car_accidents_downsampledtrain_test["LIGHTCOND"].replace("Dark - Unknown Lighting", 2, inplace=True)


#Encoding ADDRTYPE 
car_accidents_downsampledtrain_test["ADDRTYPE"].replace("Block", 0, inplace=True)
car_accidents_downsampledtrain_test["ADDRTYPE"].replace("Intersection", 1, inplace=True)
car_accidents_downsampledtrain_test.head()

#Encoding Severitycode (0 = 1, 1 = 2) to make a binary condition. 

car_accidents_downsampledtrain_test["SEVERITYCODE"].replace(1, 0, inplace=True)
car_accidents_downsampledtrain_test["SEVERITYCODE"].replace(2, 1, inplace=True)


# In[39]:


# now it is verified that all values are encoded according to rules above. 
car_accidents_downsampledtrain_test['ADDRTYPE'].value_counts().to_frame()


# In[40]:


car_accidents_downsampledtrain_test['LIGHTCOND'].value_counts().to_frame()


# In[41]:


car_accidents_downsampledtrain_test['WEATHER'].value_counts().to_frame()


# In[42]:


car_accidents_downsampledtrain_test['ROADCOND'].value_counts().to_frame()


# In[43]:


# The libraries for the modelling process are imported 
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[44]:


# the prediction models will be based on the following variables : weather and light  conditions, road conditions, type of location
# we create the indepedent and depedent vectors to feed the future models

X=car_accidents_downsampledtrain_test[["ADDRTYPE","ROADCOND","WEATHER","LIGHTCOND"]].values

y=car_accidents_downsampledtrain_test["SEVERITYCODE"].values
y[0:5]


# In[45]:


#Normalize data
#Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases:

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[46]:


#Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=4)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[47]:


#check the size of the test and train 
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[48]:


#Calculate accuracy with the best value of K provided in the  model simulation

from sklearn.neighbors import KNeighborsClassifier
k = 50

KNN = KNeighborsClassifier(n_neighbors = k).fit(X_train,np.ravel(y_train,order='C'))
KNN
yhat = KNN.predict(X_test)
yhat[0:5]

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, KNN.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[49]:


#Check Accuracy and confusion matrix. 

from sklearn.metrics import accuracy_score
print('Accuracy score for KNN = ', accuracy_score(yhat, y_test))
print('Confusion Matrix - KNN')
print(pd.crosstab(y_test.ravel(), yhat.ravel(), rownames = ['True'], colnames = ['Predicted'], margins = True))

print(classification_report(yhat,y_test))


# In[ ]:


# Display the potential values of K and check for which ones we have the best accuracy
k_range = range(1,100)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,10,20,30,40,50,100])


# In[ ]:


#Decision Tree load libraries

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score

#Define model and fit the model with the train data 

DT = DecisionTreeClassifier(criterion="entropy", max_depth=6)
DT.fit(X_train,y_train)

# Perform prediction 
yhatDT = DT.predict(X_test)

# Calculate the accuracy 

print('Accuracy score for Decision Tree = ', accuracy_score(yhatDT, y_test))

#Visualization
print('Confusion Matrix - Decision Tree')
print(pd.crosstab(y_test.ravel(), yhatDT.ravel(), rownames = ['True'], colnames = ['Predicted'], margins = True))

print(classification_report(yhatDT,y_test))



# Define function to produce the cOnfusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

 #Plot it
confusion_matrix = confusion_matrix(y_test, yhatDT, labels=[1,0])
np.set_printoptions(precision=2)


# Plot confusion matrix
plt.figure()
plot_confusion_matrix(confusion_matrix , classes=['Injury=1','Property Damage=0'],normalize= False,  title='Confusion matrix')



# In[54]:


#Support Vector Machine

# Define the model 
from sklearn import svm
clf = svm.SVC(kernel='rbf')
# train the model 
clf.fit(X_train, np.ravel(y_train,order='C'))

# Predict the model 
yhat = clf.predict(X_test)
yhat [0:5]

from sklearn.metrics import classification_report, confusion_matrix
import itertools


# define Conusion matrix plotting algorithim 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Injury=1','Property Damage=0'],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')

#from sklearn.metrics import jaccard_similarity_score
#jaccard_similarity_score(y_test, yhat)



# In[56]:


#Logistic Regression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhatLR = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

print(log_loss(y_test, yhat_prob))

print ("Accuracy", accuracy_score(yhatLR,y_test))
print (classification_report(y_test, yhatLR))

cnf_matrix = confusion_matrix(y_test, yhatLR, labels=[1,0])
np.set_printoptions(precision=2)


        # Plot confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Injury=1','Property Damage=0'],normalize= False,  title='Confusion matrix')


# In[44]:


get_ipython().system('pip install folium')


# In[81]:


#Folium Map
import folium 
#Make reduced df from feature_df to get a few random points to make map
from folium.plugins import MarkerCluster
from folium import plugins
limit = 100000
reduced_df = car_accidents_downsampled.iloc [0:limit:5, 0:]

#Folium Map
# let's start again with a clean copy of the map of San Francisco
seattle_map = folium.Map(location=[47.61536892, -122.3302243], zoom_start=12)

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(reduced_df.Y, reduced_df.X, reduced_df.SEVERITYCODE):
    folium.Marker(
    location=[lat, lng],
    icon=None,
    popup=label,
    ).add_to(incidents)

seattle_map.add_child(incidents)

# display map
seattle_map


# 

# In[116]:


from folium.plugins import MarkerCluster
from folium import plugins

seattle_map = folium.Map(location = [47.61536892, -122.3302243], zoom_start = 12)
# let's see the accidents with personal injury  only 

df_incidents = car_accidents_downsampled[car_accidents_downsampled.SEVERITYCODE == 2]
limit = 54544
df_incidents = car_accidents_downsampled.iloc[0:limit, :]
# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X,df_incidents.SEVERITYCODE):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map

seattle_map


# In[119]:


from folium.plugins import MarkerCluster
from folium import plugins

seattle_map = folium.Map(location = [47.61536892, -122.3302243], zoom_start = 12)
# let's see the accidents with personal injury  only 

df_incidents = car_accidents_downsampled[car_accidents_downsampled.SEVERITYCODE == 1]
limit = 54544
df_incidents = car_accidents_downsampled.iloc[0:limit, :]
# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X,df_incidents.SEVERITYCODE):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map

seattle_map


# In[ ]:


from folium.plugins import MarkerCluster
from folium import plugins

seattle_map = folium.Map(location = [47.61536892, -122.3302243], zoom_start = 12)
# let's see the accidents with personal injury  only 

df_incidents = car_accidents_downsampled[car_accidents_downsampled.SEVERITYCODE == 2]
limit = 54544
df_incidents = car_accidents_downsampled.iloc[0:limit, :]
# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X,df_incidents.SEVERITYCODE):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map

seattle_map


# In[ ]:


from folium.plugins import MarkerCluster
from folium import plugins

seattle_map = folium.Map(location = [47.61536892, -122.3302243], zoom_start = 12)
# let's see the accidents with personal injury  only 

df_incidents = car_accidents_downsampled[car_accidents_downsampled.SEVERITYCODE == 1]
limit = 54544
df_incidents = car_accidents_downsampled.iloc[0:limit, :]
# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X,df_incidents.SEVERITYCODE):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map

seattle_map


# In[ ]:


8182


# In[ ]:




