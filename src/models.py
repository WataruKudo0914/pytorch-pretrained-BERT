from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class AbstractRegressor(object):
    def __init__(self,X,y):
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(X,y, test_size=0.3, random_state=42)
        self.regressor = None

    def show_result(self):
        reg = self.regressor
        reg.fit(self.X_train,self.y_train)
        pred = reg.predict(self.X_test)
        plt.scatter(pred,self.y_test)
        plt.show()
        print(r2_score(y_pred=pred,y_true=self.y_test))
        return None

class LinearRegressor(AbstractRegressor):
    def __init__(self,X,y):
        super().__init__(X,y)
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()

class RidgeReg(AbstractRegressor):
    def __init__(self,X,y):
        super().__init__(X,y)
        from sklearn.linear_model import RidgeCV
        self.regressor = RidgeCV(alphas=[0.1, 1.0, 10.0],cv=3)

class MLPRegressor(AbstractRegressor):
    def __init__(self,X,y):
        super().__init__(X,y)
        from sklearn.neural_network import MLPRegressor
        self.regressor = MLPRegressor()

class RandomForest(AbstractRegressor):
    def __init__(self,X,y):
        super().__init__(X,y)
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor()
        self.regressor = GridSearchCV(estimator=rf,param_grid = self.params(),
                          cv = 2, n_jobs = -1)

    def params(self):
        param_grid = {
            'bootstrap': [True],
            'max_depth': [3,5,10, 50, 100],
            'max_features': [2, 5],
            'min_samples_leaf': [5, 10,50],
            'min_samples_split': [3,10, 20],
            'n_estimators': [10, 300,500,1000]
        }
        return param_grid
