import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file,'rb') as f_in:
    (dv,model) = pickle.load(f_in)

app = Flask('rl_game_predict')

@app.route('/predict',methods=['POST'])

def predict():
    game = request.get_json()
    X_game = dv.transform(game)
    X = xgb.DMatrix(X_game, feature_names=dv.get_feature_names_out().tolist())
    y_pred = model.predict(X)[0]
    game_win = y_pred > 0.5

    result = {
        'game_win_probability': float(y_pred),
        'game_win_guess': bool(game_win)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)