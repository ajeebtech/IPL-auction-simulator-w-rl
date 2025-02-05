import torch
import gym
import time

start = time.perf_counter()

def mse_score(tensor1, tensor2, alpha=None):
    mse = torch.mean((tensor1 - tensor2) ** 2)
    if alpha is None:
        alpha = 1 / (mse.item() + 1e-6)  # Avoid division by zero
    return torch.exp(-alpha * mse)


def position_score(player1_bataverages, player2_bataverages, player1_bowlaverages,player2_bowlaverages):
    best_position = torch.argmax(player2_bataverages).item()
    p1_avg = player1_bataverages[best_position]
    p2_avg = player2_bataverages[best_position]
    if p2_avg > p1_avg:
        return 1.0
    dot_product = torch.dot(player1_bataverages, player2_bataverages)
    norm_p1 = torch.norm(player1_bataverages, p=2)
    norm_p2 = torch.norm(player2_bataverages, p=2)
    similarity = dot_product / (norm_p1 * norm_p2 + 1e-6)
    score = torch.sigmoid(similarity)
    print(f'post. {score}')
    return score.item()

def style_tokenizer(style):
    if style == 'right-arm offbreak' or 'slow left-arm orthodox ':
        token = 1.0
    elif style == 'legbreak googly' or 'legbreak' or 'left-arm wrist-spin':
        token = 1.5
    elif style == 'left-arm fast' or 'right-arm fast':
        token = 5
    elif style == 'left-arm medium-fast' or 'right-arm medium-fast':
        token = 4.5
    elif style == 'right-arm fast-medium' or 'left-arm fast-medium':
        token = 3
    elif style == 'right-arm medium' or 'left-arm medium' or 'right-arm slow-medium':
        token = 2.5
    else:
        token = 0.0
    return token

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

df = pd.read_csv('/Users/jatin/Documents/python/the big thing/csvs/csk_dataset.csv')
pricesdf = pd.read_csv('/Users/jatin/Documents/python/the big thing/CatBoostPredictedIPLPlayerPrices.csv')
class xPlayer(gym.Env):
    def __init__(self, tensor, name, bat_style, bowl_style, bowling_average, batting_average,
                 bowling_innings, economy_rate, field_style, awards, runs, sixes, fours, 
                 strike_rate, importance, wickets, postenbat, postenbol, batting_innings):
        self.name = name
        self.bat_style = bat_style
        self.bowl_style = bowl_style
        self.awards = awards
        self.importance = importance
        self.runs = runs
        self.bowling_innings = bowling_innings
        self.batting_innings = batting_innings
        self.batting_average = batting_average
        self.bowling_average = bowling_average
        self.economy_rate = economy_rate
        self.strike_rate = strike_rate
        self.fours = fours
        self.sixes = sixes
        self.field_style = field_style
        self.wickets = wickets
        self.tensor = tensor
        self.postenbat = postenbat
        self.postenbol = postenbol

def create_squad(df,squad={}):

    for _, row in df.iterrows():
        row['bowl_style'] = style_tokenizer(row['bowl_style'])
        row['field_style'] = 1.0 if row['field_style'] == 'wicketkeeper' else 0
        row['bat_style'] = 1.5 if row['bat_style'] == 'left-hand bat' else 1.0

        # Compute additional features
        bpi = 0 if not row['batting_innings'] else (row['fours'] + row['sixes']) / row['batting_innings']
        rpi = 0 if not row['batting_innings'] else row['runs'] / row['batting_innings']
        wpi = 0 if not row['bowling_innings'] else row['wickets'] / row['bowling_innings']

        # Create tensors
        additional_tensor = torch.tensor([bpi, rpi, wpi], dtype=torch.float32)
        selected_columns = row[['bat_style', 'bowl_style', 'field_style', 
                                'batting_average', 'strike_rate', 
                                'bowling_average', 'economy_rate']].to_numpy(dtype=float)
        tensor = torch.tensor(selected_columns, dtype=torch.float32)
        tensor = torch.cat((tensor, additional_tensor))

        # Position tensors
        position_tensor_batting = torch.tensor([
            row.get(f'{i+1}st position batting', 0.0) for i in range(11)
        ], dtype=torch.float32)

        position_tensor_bowling = torch.tensor([
            row.get(f'{i+1}st position bowling', 0.0) for i in range(11)
        ], dtype=torch.float32)

        # Create Player object
        squad[row['name']] = xPlayer(
            tensor=tensor, name=row['name'], bat_style=row['bat_style'],
            bowl_style=row['bowl_style'], batting_average=row['batting_average'],
            bowling_average=row['bowling_average'], economy_rate=row['economy_rate'],
            field_style=row['field_style'], awards=row['awards'], runs=row['runs'],
            sixes=row['sixes'], fours=row['fours'], strike_rate=row['strike_rate'],
            importance=row['importance'], wickets=row['wickets'], postenbat=position_tensor_batting,
            postenbol=position_tensor_bowling, batting_innings=row['batting_innings'],
            bowling_innings=row['bowling_innings']
        )

    return squad

def normalize_awards(player_awards, min_awards=0, max_awards=43):
    normalized_score = (player_awards - min_awards) / (max_awards - min_awards + 1e-6)
    return normalized_score

def relatability_score(newplyr, player):  # this will be done with respect to every player in that squad, this is only comparing the player to the players in the squad
    score = position_score(player2_averages=newplyr.postenbat, player1_averages=player.postenbat, player2_bowlaverages=newplyr.postenbol,player1_bowlaverages= player.postenbol)
    relatibility = mse_score(newplyr.tensor, player.tensor)
    score += relatibility
    try:
        score += normalize_awards(newplyr.awards)*0.2
    except Exception:
        pass
    try:
        score += newplyr.importance
    except:
        pass
    return score

def price_predictor(score,plyr,df=pricesdf):
    predicted_amount = 5000000 if not df[df['Player'] == plyr.name]['Predicted Amount'] else df[df['Player'] == plyr.name]['Predicted Amount']
    return predicted_amount

def budget(relatibility,predicted_price):
    if predicted_price:
        pass
def find_closest_player(isquad,newplyr):
    scores = {}
    for name,pobject in isquad:
        scores[name] = relatability_score(player = pobject,newplyr=newplyr)
    max_key = max(scores, key=scores.get)
    return max_key

def reward(price,relatibility,budget):
    # rewards based on prices at which they got bought and how useful they are
    return reward

def loot_rewards():
    # team structure
    pass
end = time.perf_counter()
elapsed = end - start
print(f"Time elapsed: {elapsed} seconds")