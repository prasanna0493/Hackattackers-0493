import pandas as pd #pd is variable of pandas
from sklearn.preprocessing import LabelEncoder #labelencoder is use for converts cetagorical data into numerical data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import joblib #to download the model

df=pd.read_csv('feeds1.csv')

label_encoder=LabelEncoder()
df['action']=label_encoder.fit_transform(df['action'])

X=df[['temperature','humidity']]
Y=df['action']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)

joblib.dump(model,'ac_model.pkl')
print("model trained and downloaded successfully")