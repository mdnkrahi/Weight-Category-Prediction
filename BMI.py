import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
data.head()

data.describe()


def give_names_to_indices(ind):
    if ind==0:
        return 'Extremely Weak'
    elif ind==1:
        return 'Weak'
    elif ind==2:
        return 'Normal'
    elif ind==3:
        return 'OverWeight'
    elif ind==4:
        return 'Obesity'
    elif ind==5:
        return 'Extremely Obese'


data['Index'] = data['Index'].apply(give_names_to_indices)

data

sns.lmplot('Height','Weight',data,hue='Index',size=7,aspect=1,fit_reg=False)

people = data['Gender'].value_counts()
people

categories = data['Index'].value_counts()
categories


# STATS FOR MEN
data[data['Gender']=='Male']['Index'].value_counts()


# STATS FOR WOMEN
data[data['Gender']=='Female']['Index'].value_counts()

data = pd.DataFrame(data)
data.head()


data2 = pd.get_dummies(data['Gender'])
data.drop('Gender',axis=1,inplace=True)
data = pd.concat([data,data2],axis=1)
data.head()


y=data['Index']
data =data.drop(['Index'],axis=1)

data.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)
data=pd.DataFrame(data)
data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=101)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


param_grid = {'n_estimators':[100,200,300,400,500,600,700,800,1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101),param_grid,verbose=3)


grid_cv.fit(X_train,y_train)


grid_cv.best_params_


pred = grid_cv.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print('Acuuracy is --> ',accuracy_score(y_test,pred)*100)
print('\n')



def lp(details):
    gender = details[0]
    height = details[1]
    weight = details[2]
    
    if gender=='Male':
        details=np.array([[np.float(height),np.float(weight),0.0,1.0]])
    elif gender=='Female':
        details=np.array([[np.float(height),np.float(weight),1.0,0.0]])
    
    y_pred = grid_cv.predict(scaler.transform(details))
    return (y_pred[0])
    

#Live predictor

your_details = ['Male',175,80]
print(lp(your_details))