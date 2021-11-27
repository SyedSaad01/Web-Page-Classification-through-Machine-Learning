from sklearn import metrics
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


#Reading Dataset
URL = 'https://drive.google.com/file/d/1FwOzu6HcwpZa31PMlZqYpmVPE10lg-YR/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
df = pd.read_csv(path,engine='python', error_bad_lines=False)

#Dropping the columns whose correlation with output feature is weak
#The reason to drop content feature is that it requires NLP technique to deal so for this project we are restricting ourselves till ML techniques.
df=df.drop(['Unnamed: 0','url', 'ip_add', 'content'], axis=1)

#Dealing with outliers for js_length feature
min_threshold, max_threshold = df.js_len.quantile([0.001, 0.99])
df.js_len = np.where((df.js_len <  min_threshold) | (df.js_len > max_threshold), np.nan, df.js_len)
df.js_len =  df.js_len.fillna(df.js_len.max())

#Dealing with outliers for js_obf_length feature
min_threshold, max_threshold = df.js_obf_len.quantile([0.001, 0.99])
df.js_obf_len = np.where((df.js_obf_len <  min_threshold), min_threshold,
                     np.where((df.js_obf_len > max_threshold), max_threshold, df.js_obf_len))

#Dealing with outliers for url_length feature
min_threshold, max_threshold = df.url_len.quantile([0.001, 0.99])
df.url_len = np.where((df.url_len <  min_threshold), min_threshold,
                      np.where((df.url_len > max_threshold), max_threshold, df.url_len))
df.url_len =  df.url_len.fillna(df.url_len.max())


#Encoding of categorical features through label encoding
lb_encoders = [] 
catg_cols=['who_is',
       'https', 'label'
       ]
for i in catg_cols:
  lb_encoder = LabelEncoder()
  col = df[i]
  encoded_col = lb_encoder.fit_transform(col)
  df[i] = encoded_col
  lb_encoders.append(lb_encoder)


org=df.copy()
df2 = df

#Encoding of geolocation & tld through ranking
df2['geo_loc']=df.groupby('geo_loc')['js_obf_len'].rank(ascending=False, method = 'average')
df2['tld']=df.groupby('tld')['js_obf_len'].rank(ascending=False, method = 'average')

#Input feature
X=df2.drop(['label'], axis=1)

#output feature
y=df2['label']


#Dictionary for geolocation
dic_for_geolocation={}
for indexs,i in enumerate(org['geo_loc']):
  if i not in dic_for_geolocation.keys():
      dic_for_geolocation[i]=df2.iloc[indexs,1]



#Dictionary for top_level_domain
dic_for_tld={}
for indexs,i in enumerate(org['tld']):
  if i not in dic_for_tld.keys():
      dic_for_tld[i]=df2.iloc[indexs,2]


#Scaling all the numeric columns between 0 & 1
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))


#Since dataset is imbalance dataset so using SMOTE technique
smote = SMOTE()
X_oversampled, y_oversampled = smote.fit_resample(X, y)
X_oversampled = pd.DataFrame(X_oversampled, columns=X.columns)
y_oversampled = pd.Series(y_oversampled)


#Dividing dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.25, random_state=42)



#Applying Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_rf = rf.predict(X_test)


#Storing model  
pickle.dump(rf, open("model.sav", "wb"))
#Storing scaler object
pickle.dump(scaler, open("scaling.sav", "wb"))
#Storing geolocation dictionary
pickle.dump(dic_for_geolocation, open("geolocation_dictionary.sav", "wb"))
#Storing tld dictionary
pickle.dump(dic_for_tld, open("tld_dictionary.sav", "wb"))

#Accuracy on test dataset
print("The accuracy is "+str(metrics.accuracy_score(y_test,y_pred_rf)*100)+"%")
                     


