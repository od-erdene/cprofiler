import joblib
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, log_loss

################ reading the  model from file system ##################
filename = "./gender_prediction_histGB_May_01.sav" # model path to load
loaded_model = joblib.load(filename)

################ Here is prediction #####################
"""
input:  2d array, i.e, (n, len(columns_to_use))
output: 1d array, i.e, (n, )                    -> array([1,0,1, ...])
"""
y_pred = loaded_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('log loss:', log_loss(y_test, y_pred))

class Model1(object):
    def __init__(self):
        self._model = joblib.load("./params/gender_prediction_histGB_May_01.sav")

    def predict(self, fl):
