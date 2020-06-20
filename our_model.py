from datetime import datetime
import pandas as pd
from pandas.api.types import is_numeric_dtype


# https://machinelearningmastery.com/feature-selection-machine-learning-python/
from random_forest import random_forest


class our_model:
    def __init__(self):
        self.country_dataframe = None
        self.league_dataframe = None
        self.match_dataframe = None
        self.player_dataframe = None
        self.player_attributes_dataframe = None
        self.team_dataframe = None
        self.team_attributes_dataframe = None
        self.df = pd.DataFrame()
        self.test = pd.DataFrame()
        self.training = pd.DataFrame()

    def read_data(self, path):
        self.match_dataframe = pd.read_csv(path + "Match.csv")
        self.team_attributes_dataframe = pd.read_csv(path + "Team_Attributes.csv")
        self.player_attributes_dataframe = pd.read_csv(path + "Player_Attributes.csv")
        self.player_attributes_dataframe = self.player_attributes_dataframe.groupby('player_api_id').mean()


    def define_datafram(self):
        # for genetic
        self.df = self.match_dataframe
        self.df = self.df.dropna(subset=['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4',
                                         'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8',
                                         'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                                         'away_player_2', 'away_player_3', 'away_player_4',
                                         'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8',
                                         'away_player_9', 'away_player_10', 'away_player_11', ])
        self.df['result'] = self.calculate_win_or_loss()

        self.fill_na()



        self.test = self.df.loc[self.df["season"] == '2015/2016']
        self.testing_feature = pd.DataFrame()
        self.add_number_win_loss_five_game(self.testing_feature, self.test)
        self.add_defence_attack(self.testing_feature,  self.test)
        self.define_player(self.testing_feature, self.test)

        self.training = self.df.loc[self.df["season"] != '2015/2016']
        self.training_feature = pd.DataFrame()
        self.add_number_win_loss_five_game(self.training_feature, self.training)
        self.add_defence_attack(self.training_feature, self.training)
        self.define_player(self.training_feature, self.training)

        self.training_feature.to_excel("training_feature_all.xlsx")
        self.training.to_excel("training_all.xlsx")
        self.testing_feature.to_excel("testing_feature_all.xlsx")
        self.test.to_excel("test_all.xlsx")

        # randomForest = random_forest()
        # randomForest.main(self.training_feature, self.training['result'], self.testing_feature, self.test['result'])

    def define_player(self, dataframe, data):
        player_home_1 = list()
        player_home_2 = list()
        player_home_3 = list()
        player_home_4 = list()
        player_home_5 = list()
        player_home_6 = list()
        player_home_7 = list()
        player_home_8 = list()
        player_home_9 = list()
        player_home_10 = list()
        player_home_11 = list()

        player_away_1 = list()
        player_away_2 = list()
        player_away_3 = list()
        player_away_4 = list()
        player_away_5 = list()
        player_away_6 = list()
        player_away_7 = list()
        player_away_8 = list()
        player_away_9 = list()
        player_away_10 = list()
        player_away_11 = list()

        for index, row in data.iterrows():
            h_1 = self.player_attribute(row['home_player_1'])
            player_home_1.append(h_1)
            h_2 = self.player_attribute(row['home_player_2'])
            player_home_2.append(h_2)
            h_3 = self.player_attribute(row['home_player_3'])
            player_home_3.append(h_3)
            h_4 = self.player_attribute(row['home_player_4'])
            player_home_4.append(h_4)
            h_5 = self.player_attribute(row['home_player_5'])
            player_home_5.append(h_5)
            h_6 = self.player_attribute(row['home_player_6'])
            player_home_6.append(h_6)
            h_7 = self.player_attribute(row['home_player_7'])
            player_home_7.append(h_7)
            h_8 = self.player_attribute(row['home_player_8'])
            player_home_8.append(h_8)
            h_9 = self.player_attribute(row['home_player_9'])
            player_home_9.append(h_9)
            h_10 = self.player_attribute(row['home_player_10'])
            player_home_10.append(h_10)
            h_11 = self.player_attribute(row['home_player_11'])
            player_home_11.append(h_11)

            a_1 = self.player_attribute(row['away_player_1'])
            player_away_1.append(a_1)
            a_2 = self.player_attribute(row['away_player_2'])
            player_away_2.append(a_2)
            a_3 = self.player_attribute(row['away_player_3'])
            player_away_3.append(a_3)
            a_4 = self.player_attribute(row['away_player_4'])
            player_away_4.append(a_4)
            a_5 = self.player_attribute(row['away_player_5'])
            player_away_5.append(a_5)
            a_6 = self.player_attribute(row['away_player_6'])
            player_away_6.append(a_6)
            a_7 = self.player_attribute(row['away_player_7'])
            player_away_7.append(a_7)
            a_8 = self.player_attribute(row['away_player_8'])
            player_away_8.append(a_8)
            a_9 = self.player_attribute(row['away_player_9'])
            player_away_9.append(a_9)
            a_10 = self.player_attribute(row['away_player_10'])
            player_away_10.append(a_10)
            a_11 = self.player_attribute(row['away_player_11'])
            player_away_11.append(a_11)

        dataframe['home_player_1'] = player_home_1
        dataframe['home_player_2'] = player_home_2
        dataframe['home_player_3'] = player_home_3
        dataframe['home_player_4'] = player_home_4
        dataframe['home_player_5'] = player_home_5
        dataframe['home_player_6'] = player_home_6
        dataframe['home_player_7'] = player_home_7
        dataframe['home_player_8'] = player_home_8
        dataframe['home_player_9'] = player_home_9
        dataframe['home_player_10'] = player_home_10
        dataframe['home_player_11'] = player_home_11

        dataframe['away_player_1'] = player_away_1
        dataframe['away_player_2'] = player_away_2
        dataframe['away_player_3'] = player_away_3
        dataframe['away_player_4'] = player_away_4
        dataframe['away_player_5'] = player_away_5
        dataframe['away_player_6'] = player_away_6
        dataframe['away_player_7'] = player_away_7
        dataframe['away_player_8'] = player_away_8
        dataframe['away_player_9'] = player_away_9
        dataframe['away_player_10'] = player_away_10
        dataframe['away_player_11'] = player_away_11

    def player_attribute(self, id_player):
        # for index, row in self.player_attributes_dataframe.iterrows():
        #     if int(row['player_api_id']) == id_player:
        row = self.player_attributes_dataframe.loc[id_player, :]
        return int(row['gk_reflexes']) + int(row['gk_positioning']) + \
               int(row['gk_kicking']) \
               + int(row['gk_handling']) + int(row['gk_diving']) + \
               int(row['sliding_tackle']) + \
               int(row['standing_tackle']) + int(row['marking']) + \
               int(row['penalties']) + int(row['vision']) + int(row['positioning']) + \
               int(row['interceptions']) \
               + int(row['aggression']) + int(row['long_shots']) + \
               int(row['strength']) + \
               int(row['standing_tackle']) + int(row['jumping']) + \
               int(row['shot_power']) + int(row['strength']) + \
               int(row['balance']) + int(row['reactions']) + \
               int(row['agility']) + \
               int(row['sprint_speed']) + int(row['acceleration']) + \
               int(row['ball_control']) + int(row['long_passing']) + \
               int(row['free_kick_accuracy']) + int(row['free_kick_accuracy']) + \
               int(row['curve']) + \
               int(row['dribbling']) + \
               int(row['volleys']) + int(row['crossing']) + \
               int(row['heading_accuracy']) + int(row['finishing']) + \
               int(row['crossing']) + int(row['potential']) + \
               int(row['overall_rating'])



    def convert_data(self):
        newcol = list()
        for i in self.df['date']:
            index = i.find(" ")
            i = i[:index]
            # date = datetime.strptime(i, '%Y-%m-%d')
            date = datetime.strptime(i, '%Y-%m-%d')
            newcol.append(date)
        return newcol

    def add_defence_attack(self, dataframe, data):
        defence_home = list()
        attack_home = list()
        defence_away = list()
        attack_away = list()
        for index, row in data.iterrows():
            i, j = self.defence_attack(row['home_team_api_id'])
            defence_home.append(i)
            attack_home.append(j)
            h, l = self.defence_attack(row['away_team_api_id'])
            defence_away.append(h)
            attack_away.append(l)

        dataframe['defence_home'] = defence_home
        dataframe['attack_home'] = attack_home
        dataframe['defence_away'] = defence_away
        dataframe['attack_away'] = attack_away

    def defence_attack(self, id_team):
        sum_attack = 0
        sum_defence = 0
        for index, row in self.team_attributes_dataframe.iterrows():
            if int(row['team_api_id']) == id_team:
                sum_attack = int(row['buildUpPlaySpeed']) + int(row['buildUpPlayDribbling']) + \
                             int(row['buildUpPlayPassing']) \
                             + int(row['chanceCreationPassing']) + int(row['chanceCreationCrossing']) + \
                             int(row['chanceCreationShooting'])
                sum_defence = int(row['defencePressure']) + int(row['defenceAggression']) + \
                              int(row['defenceTeamWidth'])
                break
        return sum_defence, sum_attack

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
            h, l, m = self.find_five_game(row['away_team_api_id'], row['date'], 0, data)
            x4.append(h)
            x6.append(l)
            x8.append(m)
            q, w, e = self.find_five_game(row['home_team_api_id'], row['date'], 1, data)
            x11.append(w)
            x12.append(e)
            x13.append(k)
            r, t, y = self.find_five_game(row['away_team_api_id'], row['date'], -1, data)
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
        for column in self.team_attributes_dataframe:
            if is_numeric_dtype(self.team_attributes_dataframe[column].dtype) and column != 'team_api_id':
                self.team_attributes_dataframe[column].fillna(self.team_attributes_dataframe[column].mean(), inplace=True)
        for column in self.player_attributes_dataframe:
            if is_numeric_dtype(self.player_attributes_dataframe[column].dtype) and column != 'player_api_id':
                self.player_attributes_dataframe[column].fillna(self.player_attributes_dataframe[column].mean(),
                                                                inplace=True)


data = our_model()
data.read_data("C:/Users/TMP467/Desktop/data mimi project/")
data.define_datafram()


