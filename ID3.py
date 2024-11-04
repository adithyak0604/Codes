import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r"/home/adithyak06/Documents/Visual Code/Python Lab/ID2.csv")
enc=LabelEncoder()
df_num_cat=pd.DataFrame()
df_num_cat["Outlook"]=enc.fit_transform(df["Outlook"])
df_num_cat["Temperature"]=enc.fit_transform(df["Temperature"])
df_num_cat["Humidity"]=enc.fit_transform(df["Humidity"])
df_num_cat["Wind"]=enc.fit_transform(df["Wind"])
df_num_cat["PlayTennis"]=enc.fit_transform(df["PlayTennis"])
x=df_num_cat.drop(["PlayTennis"],axis=1)
y=df_num_cat["PlayTennis"]
from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier(criterion='entropy')
dt_clf.fit(x,y)
dt_clf.score(x,y)
y_pred=dt_clf.predict(x)
import sklearn.metrics
lbs=[0,1]
cf=sklearn.metrics.confusion_matrix(y,y_pred,labels=lbs)
acc=sklearn.metrics.accuracy_score(y,y_pred)
pre=sklearn.metrics.precision_score(y,y_pred,labels=lbs,pos_label=0)
re=sklearn.metrics.recall_score(y,y_pred,labels=lbs,pos_label=0)
r=dt_clf.predict([[0,1,1,1]])
print("confusion_matrix:\n",cf)
print("accuracy",acc)
print("precision",pre)
print("recall",re)
print("class",r) 