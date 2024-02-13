#importing required libraries 
 
from flask import Flask, request, render_template 
import numpy as np 
import pandas as pd 
from sklearn import metrics 
import warnings import pickle warnings.filterwarnings('ignore') from feature import FeatureExtraction 
 
file = open("pickle/model.pkl","rb") gbc = pickle.load(file) file.close() 
 
 
app = Flask(__name__) 
 
@app.route("/", methods=["GET", "POST"]) def index(): 
if request.method == "POST": 
 
   url = request.form["url"] 
   obj = FeatureExtraction(url) 
   x =np.array(obj.getFeaturesList()).reshape(1,30) 
 
y_pred =gbc.predict(x)[0] 
#1 is safe 
#-1 is unsafe y_pro_phishing =gbc.predict_proba(x)[0,0] y_pro_non_phishing   = gbc.predict_proba(x)[0,1]
# if(y_pred ==1 ): 
pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100) 
return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url ) return render_template("index.html", xx =-1) 
 
if __name__ == "__main__": 
app.run(debug=True) 
 
 
Comparison of Models 
 
result = pd.DataFrame({ 'ML Model' : ML_Model, 
'Accuracy' : accuracy, 
'f1_score' : f1_score, 
'Recall' : recall, 
'Precision': precision, 
}) 
 
 
 
Implementation of best ML model 
 
# XGBoost Classifier Model 
from xgboost import XGBClassifier 
 
# instantiate the model 
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7) 
 
# fit the model 
gbc.fit(X_train,y_train) 
 
 
import pickle 
 
# dump information to that file 
pickle.dump(gbc, open('pickle/model.pkl', 'wb')) 
 
#checking the feature improtance in the model 
plt.figure(figsize=(9,7)) n_features = X_train.shape[1] 
plt.barh(range(n_features), gbc.feature_importances_, align='center') plt.yticks(np.arange(n_features), X_train.columns) 
plt.title("Feature importances using permutation on full model") plt.xlabel("Feature importance") 
plt.ylabel("Feature")  
plt.show()
