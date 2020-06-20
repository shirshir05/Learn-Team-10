from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

class random_forest:

    def main(self, data_x, data_y, test_x, test_y):
        # define dataset
        self.X = data_x
        self.y = data_y
        self.test_x = test_x
        self.test_y = test_y
        # get the models to evaluate
        models = self.get_models
        # evaluate the models and store results
        results1,   results2, results3, names = list(), list(),  list(), list()
        for name, model in models.items():
            score_1, socre_2, accuracy_model = self.evaluate_model(model, name)
            results1.append(score_1)
            results2.append(socre_2)
            results3.append(accuracy_model)
            names.append(name)
        # plot model performance for comparison
        pyplot.boxplot(results1,  showmeans=True)
        pyplot.xticks(rotation=45)
        pyplot.show()
        pyplot.boxplot(results2,  showmeans=True)
        pyplot.xticks(rotation=45)
        pyplot.show()
        pyplot.boxplot(results3,  showmeans=True)
        pyplot.xticks(rotation=45)
        pyplot.show()

    # get a list of models to evaluate
    @property
    def get_models(self):
        models = dict()
        models['10'] = RandomForestClassifier(criterion='gini', max_depth=2)
        models['10'].fit(self.X, self.y)
        models['20'] = RandomForestClassifier(criterion='entropy', max_depth=2)
        models['20'].fit(self.X, self.y)
        models['30'] = RandomForestClassifier(criterion='gini', max_depth=50)
        models['30'].fit(self.X, self.y)
        models['40'] = RandomForestClassifier(criterion='entropy', max_depth=50)
        models['40'].fit(self.X, self.y)
        models['50'] = RandomForestClassifier(criterion='gini', max_depth=500)
        models['50'].fit(self.X, self.y)
        models['60'] = RandomForestClassifier(criterion='entropy', max_depth=500)
        models['60'].fit(self.X, self.y)
        models['70'] = RandomForestClassifier(criterion='entropy', max_depth=250)
        models['70'].fit(self.X, self.y)
        models['80'] = RandomForestClassifier(criterion='gini', max_depth=250)
        models['80'].fit(self.X, self.y)
        models['90'] = RandomForestClassifier(criterion='entropy', max_depth=10)
        models['90'].fit(self.X, self.y)
        models['100'] = RandomForestClassifier(criterion='gini', max_depth=10)
        models['100'].fit(self.X, self.y)
        models['110'] = RandomForestClassifier(criterion='entropy', max_depth=5)
        models['110'].fit(self.X, self.y)
        models['120'] = RandomForestClassifier(criterion='gini', max_depth=5)
        models['120'].fit(self.X, self.y)
        models['130'] = RandomForestClassifier(criterion='entropy', max_depth=1)
        models['130'].fit(self.X, self.y)
        models['140'] = RandomForestClassifier(criterion='gini', max_depth=1)
        models['140'].fit(self.X, self.y)
        return models

    # evaluate a give model using cross-validation
    def evaluate_model(self, model, name):
        print(" 1- f1 score for GP - name: " + name)
        pre = model.predict(self.test_x)

        score_1 = f1_score(self.test_y, pre, average='macro')
        print(score_1)
        print(" 1.2- f1 score for GP None: ")
        print(f1_score(self.test_y, pre, average=None))

        # Calculate the accuracy
        print("2 - accuracy score for GP: ")
        socre_2 = accuracy_score(self.test_y, pre, normalize=True)
        print(socre_2)

        # KFold Cross Validation approach
        kf = KFold(n_splits=10, shuffle=False)
        kf.split(self.test_x)

        # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
        accuracy_model = []
        index = 1
        # Iterate over each train-test split
        for train_index, test_index in kf.split(self.test_x):
            # Split train-test
            X_train, X_test = self.test_x.iloc[train_index], self.test_x.iloc[test_index]
            y_train, y_test = self.test_y.iloc[train_index], self.test_y.iloc[test_index]
            # Train the model
            model.fit(X_train, y_train)
            # Append to accuracy_model the accuracy of the model
            acc = accuracy_score(y_test, model.predict(X_test), normalize=True) * 100
            print("K-fold number %d, accuracy_score %d", index, acc)
            accuracy_model.append(acc)
            index += 1

        # Print the accuracy
        print("3 - K-Fold Cross Validation for GP: ")
        print(accuracy_model)

        return score_1, socre_2, accuracy_model

