import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report
import argparse

# from tensorflow import keras

def extra_features(data):
    # datetime features
    df = data.copy()
    df['day']       = df['date'].dt.day
    df['month']     = df['date'].dt.month
    df['year']      = df['date'].dt.year
    df['day_week']  = df['date'].dt.weekday
    df['weekmonth'] = (df['day'] - 1) // 7 + 1
    
    # devices features
    df['sector']     = df['device'].str[:4]
    df['equipment']  = df['device'].str[4:]
    return df

def multi(x,i):
    if x%i == 0:
        return True
    else:
        return False
    


def preprocess(X,Y,OHE,low_card,scaler,i,sm):
    x = X.copy()
    y = Y.copy()
    
    x.reset_index(inplace = True, drop = True)
    y.reset_index(inplace = True, drop = True)
    
    if i == 1:
        
        coding_hot = OHE.fit_transform(x[low_card].to_numpy())
        aux = pd.DataFrame(coding_hot, columns = OHE.get_feature_names_out(low_card))
        x = pd.concat([x, aux], axis=1).drop(low_card, axis=1)
        
        aux = scaler.fit_transform(x)
        x = pd.DataFrame(aux,index=x.index, columns=x.columns)
        xf, yf = sm.fit_resample(x, y) 
        
    else:
        # encoding the categorical variables
        coding_hot = OHE.fit_transform(x[low_card].to_numpy())
        aux = pd.DataFrame(coding_hot, columns = OHE.get_feature_names_out(low_card))
        x = pd.concat([x, aux], axis=1).drop(low_card, axis=1)
        
        aux2 = scaler.fit_transform(x)
        xf = pd.DataFrame(aux2,index=x.index, columns=x.columns)
        yf = y
    
    return xf, yf


def processing(df, model_path):
  df['date'] = pd.to_datetime(df['date']) 
  df = extra_features(df)
  a = df['metric1'].apply(lambda x: multi(x,8))
  df['mnw1'] = df['metric1']/8
  df['mnw1'] = df['mnw1'].apply(lambda x: math.ceil(x))
  df[df['metric2'] == 55].metric2.count()
  idx = df['metric2'] == 55
  df.loc[idx,'metric2'] = 56
  df['mnw2'] = df['metric2']/8
  df['mnw2'] = df['mnw2'].astype(int)

  df['metric7'].sort_values().unique()
  df['mnw7'] = df['metric7']/2
  df['mnw7'] = df['mnw7'].astype(int)

  df['metric8'].sort_values().unique()
  df['mnw8'] = df['metric8']/2
  df['mnw8'] = df['mnw8'].astype(int)
  df.drop(['metric1','metric2','metric7','metric8'], axis = 1 , inplace = True)


  df['dif_m6'] = df['metric6']
  df['dif_m5'] = df['metric5']

  df['log_m2'] = np.log(df['mnw2']+1)
  df['log_m3'] = np.log(df['metric3']+1)
  df['log_m4'] = np.log(df['metric4']+1)
  df['log_m7'] = np.log(df['mnw7']+1)
  df['log_m8'] = np.log(df['mnw8']+1)
  df['log_m9'] = np.log(df['metric9']+1)

  df.drop(['mnw2','metric3','metric4','mnw7','mnw8','metric9'], axis = 1, inplace = True)
  dev_name = df['device'].unique()

  for i in dev_name:
      filt = df[df['device'] == i]
      df.loc[filt.index,'dif_m6'] = filt['dif_m6'] - filt['metric6'].min()
      df.loc[filt.index,'dif_m5'] = filt['dif_m5'] - filt['metric5'].min()

  df_cod = df.drop(['failure','equipment','device','date','year'],axis =1)
  df_final = df.drop(['device','log_m8','weekmonth','year','equipment','date','month'],axis =1)

  X = df_final.drop(['failure'], axis = 1)
  y = df_final['failure']
  # X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state = 0, test_size=0.25, shuffle = True)
  # X_train.reset_index(inplace = True, drop = True)
  # Y_train.reset_index(inplace = True, drop = True)

  # X_test.reset_index(inplace = True, drop = True)
  # Y_test.reset_index(inplace = True, drop = True)

  # x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, random_state = 0, test_size=0.25)

  low_card =['sector']

  OHE =  OneHotEncoder(handle_unknown = 'ignore',sparse=False)

  scaler = StandardScaler()
  #scaler = MinMaxScaler()

  sm = SMOTE(random_state=0)

  # X_res, Y_res = preprocess(x_train, y_train,OHE,low_card,scaler,1,sm)   
  # X_val, Y_val = preprocess(x_val, y_val,OHE,low_card,scaler,0,sm)

  X_val, Y_val = preprocess(X, y,OHE,low_card,scaler,0,sm)


  # X = np.expand_dims(X_res, axis=1)

  X_val = np.expand_dims(X_val, axis=1)


  X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])

  # X = X.reshape(X.shape[0], X.shape[2], X.shape[1])

  # Load the model from the file
  #with open(model_path, 'rb') as file:
      #model = pickle.load(file)
    
  model = keras.models.load_model('model_lstm.h5')
  
  y_pred = model.predict(X_val) 


  fpr, tpr, thresholds = roc_curve(Y_val, y_pred)
  auc = roc_auc_score(Y_val, y_pred)
  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
  plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc='lower right')
  plt.show()

  ypred = np.where(y_pred >= 0.5, 1, 0)
  report=classification_report(Y_val, ypred)
#   print('ROC AUC: ',round(roc_auc_score(Y_val,y_pred),4))
#   print(report)
#   print (y_pred) #mine

#   failure_count = ypred.count(0)
#   no_failure_count = ypred.count(1)

#   # Print the counts
#   print("Failure count:", failure_count)
#   print("No failure count:", no_failure_count)

  return y_pred, Y_val, ypred, report




# if __name__ == "__main__":
#   parser = argparse.ArgumentParser(description='Description of your program')
#   parser.add_argument('-file_path','--file_path', help='Test set file path', required=True)
#   parser.add_argument('-model_path','--model_path', help='model_path', required=True)
#   args = vars(parser.parse_args())

#   df = pd.read_csv(args['file_path'])

# processing(df, args['model_path'])





  


  











