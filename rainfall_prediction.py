import pandas as pd
import numpy as np
data=pd.read_csv('austin_weather.csv')
print(data.head(2))
print(data.shape)
data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches', 'SeaLevelPressureLowInches'], axis = 1)

# some values have 'T' which denotes trace rainfall, we need to replace all occurrences of T with 0
# so that we can use the data in our model
data = data.replace('T', 0.0)
# data also contain '-'  where values are nill, we also need to remove them
data = data.replace('-', 0.0)
# save the data in a csv file
data.to_csv('austin_final.csv')
# reading the cleaned data again
data = pd.read_csv("austin_final.csv")
# here rain has been represented by the term precipitation, it will serve as the label column
X = data.drop(['PrecipitationSumInches'], axis = 1)
y= data['PrecipitationSumInches']
y= y.values.reshape(-1, 1)
# consider a random day in the dataset we shall plot a graph and observe this day

day_index = 798
days = [i for i in range(y.size)]
from sklearn.linear_model import LinearRegression
model3 = LinearRegression()
model3.fit(X, y)
y_pred=model3.predict(X)
import pickle
# Save the trained model as a pickle string.
saved_model = pickle.dumps(model3)
 
# Load the pickled model
model3_from_pickle = pickle.loads(saved_model)
 
# Use the loaded pickled model to make predictions
model3_from_pickle.predict(X)
# or we can use joblib to save our model

import joblib
 
# Save the model as a pickle in a file
joblib.dump(model3, 'model3.pkl')
 
# Load the model from the file
rainpred_from_joblib = joblib.load('model3.pkl')
 
# Use the loaded model to make predictions
print(rainpred_from_joblib.predict(X))
