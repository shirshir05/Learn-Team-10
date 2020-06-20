# https://blogs.itility.nl/en/how-to-create-a-model-that-predicts-housing-prices-in-boston-by-using-symbolic-regression

# x3 The number of win games of the home team during the last five rounds -> V
# x4 The number of draw games of the home team during the last five rounds-> V
# x5 The number of lose games of the home team during the last five rounds -> V
# x6 The number of win games of the away team during the last five rounds -> V
# x7 The number of draw games of the away team during the last five rounds -> V
# x8 The number of lose games of the away team during the last five rounds -> V

# x11 The number of win games of the home team of its last five home games -> V
# x12 The number of draw games of the home team of its last five home games -> V
# x13 The number of lose games of the home team of its last five home games -> V
# x14 The number of win games of the away team of its last five away games -> V
# x15 The number of draw games of the away team of its last five away games -> V
# x16 The number of lose games of the away team of its last five away games -> V
import numpy as np

from datetime import datetime
from pandas.api.types import is_numeric_dtype
from gplearn.genetic import SymbolicRegressor
import pandas as pd

# score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

class Data:
    def _init_(self):
        self.country_dataframe = None
        self.league_dataframe = None
        self.match_dataframe = None
        self.player_dataframe = None
        self.player_attributes_dataframe = None
        self.team_dataframe = None
        self.team_attributes_dataframe = None
        self.df = None
        self.df = pd.DataFrame()
        self.test = pd.DataFrame()
        self.training = pd.DataFrame()
        self.logger = open('resultLog','a+')

    def read_data(self, path):
        self.match_dataframe = pd.read_csv(path + "Match.csv")


    def define_datafram(self):
        self.df["season"] = self.match_dataframe['season']
        self.df["date"] = self.match_dataframe['date']
        self.df["date"] = self.convert_data()
        self.df["home_team_api_id"] = self.match_dataframe['home_team_api_id']
        self.df["away_team_api_id"] = self.match_dataframe['away_team_api_id']
        self.df["home_team_goal"] = self.match_dataframe['home_team_goal']
        self.df["away_team_goal"] = self.match_dataframe['away_team_goal']
        self.df['result'] = self.calculate_win_or_loss()
        self.fill_na()

        self.test = self.df.loc[self.df["season"] == '2015/2016']
        self.GP_testing = pd.DataFrame()
        self.add_number_win_loss_five_game(self.GP_testing, self.test)
        self.training = self.df.loc[self.df["season"] != '2015/2016']
        self.GP_training = pd.DataFrame()
        self.add_number_win_loss_five_game(self.GP_training, self.training)

        self.GP_training.to_excel("GP_training_all.xlsx")
        self.training.to_excel("training_for_GP_all.xlsx")
        self.GP_testing.to_excel("GP_testing_all.xlsx")
        self.test.to_excel("test_for_GP_all.xlsx")
        print("shir")

        # self.GP()

    def xlsl(self):
        self.GP_training = pd.read_excel('GP_training_all.xlsx', index_col=0)
        self.training = pd.read_excel('training_for_GP_all.xlsx', index_col=0)
        self.GP_testing = pd.read_excel('GP_testing_all.xlsx', index_col=0)
        self.test = pd.read_excel('test_for_GP_all.xlsx', index_col=0)


        # training_feature
        dataframe = self.GP_training
        array = dataframe.values
        X = array[0:, 0: 16]
        Y = self.training['result']
        # feature extraction
        model = LogisticRegression(solver='liblinear')
        rfe = RFE(model)
        fit = rfe.fit(X, Y)
        self.logger.write("training -  Num Features: %d" % fit.n_features_)
        self.logger.write("\n")
        self.logger.write("training - Selected Features: %s" % fit.support_)
        self.logger.write("\n")
        self.logger.write("training - Feature Ranking: %s" % fit.ranking_)
        self.logger.write("\n")
        self.GP_training = self.GP_training.iloc[:, fit.support_]
        self.GP_testing = self.GP_testing.iloc[:, fit.support_]

        self.GP()


    def convert_data(self):
        newcol = list()
        for i in self.df['date']:
            index = i.find(" ")
            i = i[:index]
            # date = datetime.strptime(i, '%Y-%m-%d')
            date = datetime.strptime(i, '%Y-%m-%d')
            newcol.append(date)
        return newcol

    def add_number_win_loss_five_game(self, dataframe, data):

        x3 = list()
        x4 = list()
        x5 = list()
        x6 = list()
        x7 = list()
        x8 = list()
        x11 = list()
        x12 = list()
        x13 = list()
        x14 = list()
        x15 = list()
        x16 = list()
        # for id_home, id_away, date in self.df['away_team_api_id'], self.df['away_team_api_id'], self.df['date']:
        for index, row in data.iterrows():
            i, j, k = self.find_five_game(row['home_team_api_id'], row['date'], 0, data)
            x3.append(i)
            x5.append(j)
            x7.append(k)
            h, l, m = self.find_five_game(row['away_team_api_id'],  row['date'], 0, data)
            x4.append(h)
            x6.append(l)
            x8.append(m)
            q, w, e = self.find_five_game(row['home_team_api_id'], row['date'], 1, data)
            x11.append(w)
            x12.append(e)
            x13.append(k)
            r, t, y = self.find_five_game(row['away_team_api_id'],  row['date'], -1, data)
            x14.append(r)
            x15.append(t)
            x16.append(y)
        dataframe['x3'] = x3
        dataframe['x4'] = x4
        dataframe['x5'] = x5
        dataframe['x6'] = x6
        dataframe['x7'] = x7
        dataframe['x8'] = x8
        dataframe['x11'] = x11
        dataframe['x12'] = x12
        dataframe['x13'] = x13
        dataframe['x14'] = x14
        dataframe['x15'] = x15
        dataframe['x16'] = x16

    # 0 - all (home and away) , 1 = home , -1 = away
    def find_five_game(self, id_team, date_game, index, data):
        # find all game of team
        game_home = None
        game_away = None
        all_games = None
        if index == 1 or index == 0:
            game_home = data.loc[data['home_team_api_id'] == id_team]
            all_games = game_home
        if index == -1 or index == 0:
            game_away = data.loc[data['away_team_api_id'] == id_team]
            all_games = game_away
        if index == 0:
            frames = [game_home, game_away]
            all_games = pd.concat(frames)
        game_last = all_games.loc[all_games['date'] < date_game]
        game_last = game_last.sort_values('date')
        five_games = game_last.tail(5)
        if five_games.shape[0] == 0:
            return 0, 0, 0
        win = 0
        loss = 0
        draw = 0
        for index, row in five_games.iterrows():
            if row['result'] == 1 and row['home_team_api_id'] == id_team:
                win += 1
            elif row['result'] == -1 and row['home_team_api_id'] != id_team:
                loss += 1
            else:
                draw += 1
        return win, loss, draw

    def calculate_win_or_loss(self):
        self.df['result'] = self.df['home_team_goal'] - self.df['away_team_goal']
        newcol = list()
        for i in self.df['result']:
            if i > 0:
                # home win
                newcol.append(1)
            elif i < 0:
                # home score
                newcol.append(-1)
            else:
                # same score
                newcol.append(0)
        return newcol

    # fill all na value in mean
    def fill_na(self):
        for column in self.df:
            if is_numeric_dtype(self.df[column].dtype) and column != 'result':
                self.df[column].fillna(self.df[column].mean(), inplace=True)


    def GP(self):
        data_training = pd.DataFrame(self.GP_training)
        # data_training.columns = data_training.columns
        target_training = pd.DataFrame(self.training['result'])
        target_training.columns = ['result']

        # symbolic regression
        function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'max', 'min')
        sr = SymbolicRegressor(population_size=50000,
                               generations=10, stopping_criteria=0.01, function_set=function_set,
                               p_crossover=0.1, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.15, p_point_mutation=0.2,
                               max_samples=0.7, verbose=1,
                               parsimony_coefficient=0.025, random_state=0)
        sr.fit(data_training, target_training)

        # check results
        # Returns the coefficient of determination R^2 of the prediction.
        self.logger.write(sr.score(data_training, target_training))
        self.logger.write("\n")

        data_test = pd.DataFrame(self.GP_testing)
        target_test = pd.DataFrame(self.test['result'])
        predict_test = sr.predict(data_test)
        pre = list()
        for i in predict_test:
            if i < 0:
                pre.append(-1)
            elif i > 0:
                pre.append(1)
            else:
                pre.append(0)
        predict_test = np.asarray(pre)

        target_test.columns = ['result']


        self.logger.write(" 1.1- f1 score for GP: ")
        self.logger.write("\n")
        self.logger.write(f1_score(target_test, predict_test, average='macro'))
        self.logger.write("\n")
        self.logger.write(" 1.2- f1 score for GP None: ")
        self.logger.write("\n")
        self.logger.write(f1_score(target_test, predict_test, average=None))
        self.logger.write("\n")

        # Calculate the accuracy
        self.logger.write("2 - accuracy score for GP: ")
        self.logger.write("\n")
        self.logger.write(accuracy_score(target_test, predict_test, normalize=True))
        self.logger.write("\n")

        # KFold Cross Validation approach
        kf = KFold(n_splits=10, shuffle=False)
        kf.split(data_test)

        # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
        accuracy_model = []
        index = 1
        # Iterate over each train-test split
        for train_index, test_index in kf.split(data_test):
            # Split train-test
            X_train, X_test = data_test.iloc[train_index], data_test.iloc[test_index]
            y_train, y_test = target_test.iloc[train_index], target_test.iloc[test_index]
            # Train the model
            sr.fit(X_train, y_train)
            # Append to accuracy_model the accuracy of the model
            predict_test = sr.predict(X_test)
            pre = list()
            for i in predict_test:
                if i < 0:
                    pre.append(-1)
                elif i > 0:
                    pre.append(1)
                else:
                    pre.append(0)
            predict_test = np.asarray(pre)
            acc = accuracy_score(y_test,predict_test, normalize=True) * 100
            self.logger.write("K-fold number %d, accuracy_score %d", index, acc)
            self.logger.write("\n")
            accuracy_model.append(acc)
            index += 1

        # Print the accuracy
        self.logger.write("3 - K-Fold Cross Validation for GP: ")
        self.logger.write("\n")
        self.logger.write(accuracy_model)
        self.logger.write("\n")


def main():
    try:
        data = Data()
        # data.read_data("C:/Users/TMP467/Desktop/data mimi project/")
        # data.define_datafram()
        data.xlsl()
    except Exception as e:
        with open('resultLog' , 'a+') as log:
            pass
            log.write(str(e))


if __name__ == "__main__":
    main()