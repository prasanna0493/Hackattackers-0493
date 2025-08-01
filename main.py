from flask import Flask,request,jsonify
import joblib
import numpy as np

app=Flask(__name__)
model=joblib.load('ac_model.pkl')

@app.route("/")
def home():
    return "AC project"

@app.route("/predict",methods=['POST'])
def predict():
    data=request.json #json java script object notation ,json is a rule to 
    temp=data.get('temperature')
    hum=data.get('humidity')

    if temp is None or hum is None:
        return jsonify({"error":"Missing temperature or humiidity"}),400
    
    prediction=model.predict(np.array([[temp,hum]]))
    status="ON" if prediction==1 else "OFF"
    return jsonify({"ac_status":status})

#server is peace of code it runs conitnuously at one place
if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)#if debug is used for automatic changes in server,if server is in running then we update code automatically change in server.