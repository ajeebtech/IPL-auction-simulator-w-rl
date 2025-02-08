import csv
from tqdm import tqdm
import re

marquee_names = []
marquee_nationalities = []
marquee_roles = []

with open('/Users/jatin/Documents/model stuff/auction2024list.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        full_name = re.sub(r'\s+', ' ', f"{row['First Name']} {row['Surname']}").strip().title()
            
        marquee_names.append(full_name)
        
        nationality = '🛺' if row['Country'] == 'India' else '✈️'
        marquee_nationalities.append(nationality)
        
        role = re.sub(r'\s+', ' ', row['Specialism']).strip().title()
        marquee_roles.append(role)


marquee = [marquee_names,marquee_nationalities,marquee_roles,20000000]

#                                                                    dataSourcing

# Chennai Super Kings 2025 Bids
top_bids_csk = ["Ravichandran Ashwin", "Noor Ahmad", "Devon Conway", "Sam Curran"]
medium_bids_csk = ["Khaleel Ahmed", "Rachin Ravindra", "Anshul Kamboj", "Rahul Tripathi"]
meh_bids_csk = ["Vijay Shankar", "Shaik Rasheed", "Kamlesh Nagarkoti", "Nathan Ellis"]
csk_bids = [top_bids_csk, medium_bids_csk, meh_bids_csk]

# Delhi Capitals 2025 Bids
top_bids_dc = ["KL Rahul", "Mitchell Starc", "T Natarajan"]
medium_bids_dc = ["Jake Fraser-McGurk", "Harry Brook", "Ashutosh Sharma"]
meh_bids_dc = ["Mohit Sharma", "Karun Nair", "Sameer Rizvi"]
dc_bids = [top_bids_dc, medium_bids_dc, meh_bids_dc]

# Gujarat Titans 2025 Bids
top_bids_gt = ["Jos Buttler", "Mohammed Siraj", "Kagiso Rabada"]
medium_bids_gt = ["Prasidh Krishna", "Washington Sundar", "Gerald Coetzee"]
meh_bids_gt = ["Manav Suthar", "Karim Janat", "Arshad Khan"]
gt_bids = [top_bids_gt, medium_bids_gt, meh_bids_gt]

# Kolkata Knight Riders 2025 Bids
top_bids_kkr = ["Venkatesh Iyer", "Anrich Nortje"]
medium_bids_kkr = ["Quinton de Kock", "Angkrish Raghuvanshi", "Vaibhav Arora"]
meh_bids_kkr = ["Manish Pandey", "Luvnith Sisodia", "Arjun Tendulkar"]
kkr_bids = [top_bids_kkr, medium_bids_kkr, meh_bids_kkr]

# Lucknow Super Giants 2025 Bids
top_bids_lsg = ["Rishabh Pant", "Avesh Khan", "Akash Deep"]
medium_bids_lsg = ["David Miller", "Mitchell Marsh", "Abdul Samad"]
meh_bids_lsg = ["Aryan Juyal", "Digvesh Singh", "Prince Yadav"]
lsg_bids = [top_bids_lsg, medium_bids_lsg, meh_bids_lsg]

# Mumbai Indians 2025 Bids
top_bids_mi = ["Trent Boult", "Deepak Chahar"]
medium_bids_mi = ["Naman Dhir", "Will Jacks", "Allah Ghazanfar"]
meh_bids_mi = ["Robin Minz", "Reece Topley", "Arjun Tendulkar"]
mi_bids = [top_bids_mi, medium_bids_mi, meh_bids_mi]

top_bids_rr = ["Sanju Samson", "Yashasvi Jaiswal", "Jofra Archer", "Wanindu Hasaranga"]
medium_bids_rr = ["Dhruv Jurel", "Shimron Hetmyer", "Trent Boult", "Ravichandran Ashwin"]
meh_bids_rr = ["Adam Zampa", "Kuldeep Sen", "Donovan Ferreira", "Kunal Singh Rathore"]
rr_bids = [top_bids_rr, medium_bids_rr, meh_bids_rr]

# Royal Challengers Bangalore 2025 Bids
top_bids_rcb = ["Virat Kohli", "Glenn Maxwell", "Josh Hazlewood", "Faf du Plessis"]
medium_bids_rcb = ["Rajat Patidar", "Mahipal Lomror", "Karn Sharma", "Reece Topley"]
meh_bids_rcb = ["Himanshu Sharma", "Suyash Prabhudessai", "Rajan Kumar", "Swapnil Singh"]
rcb_bids = [top_bids_rcb, medium_bids_rcb, meh_bids_rcb]

# Punjab Kings 2025 Bids
top_bids_pbks = ["Shikhar Dhawan", "Kagiso Rabada", "Arshdeep Singh", "Liam Livingstone"]
medium_bids_pbks = ["Jonny Bairstow", "Rahul Chahar", "Rishi Dhawan", "Harpreet Brar"]
meh_bids_pbks = ["Atharva Taide", "Vidwath Kaverappa", "Nathan Ellis", "Shivam Singh"]
pbks_bids = [top_bids_pbks, medium_bids_pbks, meh_bids_pbks]

# Sunrisers Hyderabad 2025 Bids
top_bids_srh = ["Pat Cummins", "Heinrich Klaasen", "Travis Head", "Wanindu Hasaranga"]
medium_bids_srh = ["Bhuvneshwar Kumar", "Washington Sundar", "Rahul Tripathi", "Shahbaz Ahmed"]
meh_bids_srh = ["Chetan Sakariya", "Umran Malik", "Mayank Markande", "Priyam Garg"]
srh_bids = [top_bids_srh, medium_bids_srh, meh_bids_srh]



          
class Team:
    squad = []
    def __init__(self,purse,bmen,arounders,bwlrs,overseas,wks,name,strats,squad):
        self.purse = purse
        self.bmen = bmen
        self.arounders = arounders
        self.bwlrs = bwlrs
        self.overseas = overseas
        self.wks = wks
        self.name = name
        self.strats = strats
        self.squad = squad
    
    def bid(self,amount,player):
        player.bidding(amount)
        
    def decision_making(self,player,bids):
        if player in self.strats:
            if player in self.strats[0]:
                options = ['bid','no bid']
                probabilities = [0.7,0.3]
                result = rd.choices(options,probabilities)
                if result[0] == 'bid':
                    bids[self.name] = 'bid'
                    self.bid(2000000,player)
                    # print(f'{self.name.upper()} has raised the bid by 20 lakh!,{player.price+2000000} please?')

                else:
                    pass
            elif player in self.strats[1]:
                options = ['bid','no bid']
                probabilities = [0.5,0.5]
                result = rd.choices(options,probabilities)
                if result[0] == 'bid':
                    bids[self.name] = 'bid'
                    self.bid(2000000,player)
                    # print(f'{self.name.upper()} has raised the bid by 20 lakh!,{player.price+2000000} please?')

                else:
                    pass
            elif player in self.strats[2]:
                options = ['bid','no bid']
                probabilities = [0.3,0.7]
                result = rd.choices(options,probabilities)
                if result[0] == 'bid':
                    bids[self.name] = 'bid'
                    self.bid(2000000,player)
                    # print(f'{self.name.upper()} has raised the bid by 20 lakh!,{player.price+2000000} please?')

                else:
                    pass
        else:
            options = ['bid','no bid']
            probabilities = [0.1,0.9]
            result = rd.choices(options,probabilities)
            if result[0] == 'bid':
                bids[self.name] = 'bid'
                self.bid(2000000,player)
                # print(f'{self.name.upper()} has raised the bid by 20 lakh!,{player.price+2000000} please?')

            else:
                pass

import random as rd
import time as ti
from ddpg_torch import Agent
import pandas as pd
import gym
import numpy as np

class Player:
    
    isSold = False
    def __init__(self,role,price,nationality,name): 
        self.role = role
        self.price = price
        self.nationality = nationality
        self.name = name

    def bidding(self,amount):
        self.price += amount

teams = ['csk','dc','gt','kkr','lsg','mi','pbks','rr','rcb','srh']
your_team_name = input("What team do you want to play as?(csk,dc,gt,kkr,lsg,mi,pbks,rr,rcb,srh) ")
teams_o = []         # objects of all teams
for i in range(len(teams)):
    if teams[i] == 'csk':
        csk = Team(
            purse=580000000,  # 120Cr - 62Cr retentions
            bmen=1,
            arounders=2,
            bwlrs=1,
            overseas=1,
            wks=1,
            name='csk',
            strats=csk_bids,
            squad=["Ruturaj Gaikwad", "Ravindra Jadeja", "MS Dhoni", "Shivam Dube", "Matheesha Pathirana"]
        )
        teams_o.append(csk)
    elif teams[i] == 'dc':
        dc = Team(
            purse=762500000,  # 120Cr - 43.75Cr retentions
            bmen=2,
            arounders=1,
            bwlrs=1,
            overseas=1,
            wks=1,
            name='dc',
            strats=dc_bids,
            squad=["Axar Patel", "Kuldeep Yadav", "Tristan Stubbs", "Abhishek Porel"]
        )
        teams_o.append(dc)
    elif teams[i] == 'gt':
        gt = Team(
            purse=690000000,  # 120Cr - 51Cr retentions
            bmen=3,
            arounders=1,
            bwlrs=1,
            overseas=1,
            wks=0,
            name='gt',
            strats=gt_bids,
            squad=["Rashid Khan", "Shubman Gill", "Sai Sudharsan", "Rahul Tewatia", "Shahrukh Khan"]
        )
        teams_o.append(gt)
    elif teams[i] == 'kkr':
        kkr = Team(
            purse=510000000,  # 120Cr - 69Cr retentions
            bmen=2,
            arounders=2,
            bwlrs=2,
            overseas=2,
            wks=0,
            name='kkr',
            strats=kkr_bids,
            squad=["Rinku Singh", "Varun Chakravarthy", "Sunil Narine", "Andre Russell", "Harshit Rana", "Ramandeep Singh"]
        )
        teams_o.append(kkr)
    elif teams[i] == 'lsg':
        lsg = Team(
            purse=690000000,  # 120Cr - 51Cr retentions
            bmen=2,
            arounders=0,
            bwlrs=3,
            overseas=1,
            wks=1,
            name='lsg',
            strats=lsg_bids,
            squad=["Nicholas Pooran", "Ravi Bishnoi", "Mayank Yadav", "Mohsin Khan", "Ayush Badoni"]
        )
        teams_o.append(lsg)
    elif teams[i] == 'mi':
        mi = Team(
            purse=460000000,  # 120Cr - 74Cr retentions
            bmen=3,
            arounders=1,
            bwlrs=1,
            overseas=0,
            wks=0,
            name='mi',
            strats=mi_bids,
            squad=["Jasprit Bumrah", "Suryakumar Yadav", "Hardik Pandya", "Rohit Sharma", "Tilak Varma"]
        )
        teams_o.append(mi)
    elif teams[i] == 'pbks':
        pbks = Team(
            purse=1120000000,  # 120Cr - 8Cr retentions
            bmen=2,
            arounders=0,
            bwlrs=0,
            overseas=0,
            wks=1,
            name='pbks',
            strats=pbks_bids,
            squad=["Shashank Singh", "Prabhsimran Singh"]
        )
        teams_o.append(pbks)
    elif teams[i] == 'rr':
        rr = Team(
            purse=410000000,  # 120Cr - 79Cr retentions
            bmen=2,
            arounders=1,
            bwlrs=1,
            overseas=1,
            wks=2,
            name='rr',
            strats=rr_bids,
            squad=["Sanju Samson", "Yashasvi Jaiswal", "Riyan Parag", "Dhruv Jurel", "Shimron Hetmyer", "Sandeep Sharma"]
        )
        teams_o.append(rr)
    elif teams[i] == 'srh':
        srh = Team(
            purse=697000000,  # 120Cr - 50.3Cr retentions
            bmen=2,
            arounders=1,
            bwlrs=0,
            overseas=2,
            wks=1,
            name='srh',
            strats=srh_bids,
            squad=["Heinrich Klaasen", "Pat Cummins", "Abhishek Sharma"]
        )
        teams_o.append(srh)


for i in range(len(teams)):
    if teams_o[i].name == your_team_name:
        your_team = teams_o[i]                      # i want to make a object your_team that will function as a team and have everything that teams_o object has
        teams_o.remove(teams_o[i])
        teams.remove(teams[i])
        break

print(f"Your team consists of {', '.join(your_team.squad)} and you have a purse of {your_team.purse} to make a squad of {15 - len(your_team.squad)} more competent players")
def ai(player,bids):
    for i in range(len(teams_o)):
        teams_o[i].decision_making(player,bids) #this function checks if the player is important and actively bids for the player accordingly
        pass

def adding(player,team):
    team.squad.append(player.name)
    team.purse -= player.price
    if player.role == 'Batter':
        team.bmen += 1
    elif player.role == 'Wicketkeeper':
        team.wks += 1
    elif player.role == 'Bowler':
        team.bwlrs += 1
    elif player.role == 'All Rounder':
        team.arounders += 1
    if player.nationality == "✈️":
        team.overseas += 1

def removing(player, set_container):
    """Safer removal using player reference"""
    try:
        idx = set_container[0].index(player.name)  # Find index by name
        del set_container[0][idx]
        del set_container[1][idx]
        del set_container[2][idx]
    except ValueError:
        pass
        # print(f"⚠️ {player.name} not found in set")
# retentionsdf = pd.read_csv('/Users/jatin/Documents/python/the big thing/retentions.csv')
from auction_gyms import DelhiCapitalsEnv
from processing import *
df = pd.read_csv('/Users/jatin/Documents/python/the big thing/csvs/auction_dataset.csv')
pricesdf = pd.read_csv('/Users/jatin/Documents/python/the big thing/CatBoostPredictedIPLPlayerPrices.csv')
team_df = pd.read_csv(f'/Users/jatin/Documents/python/the big thing/csvs/{your_team_name}_dataset.csv')
from auction_gyms import DelhiCapitalsEnv
env = DelhiCapitalsEnv()
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1)
isquad = create_squad(df=team_df)
np.random.seed(0)
num_iterations = 1000
def bidding(setx,obs, env,agent=agent,score=0,isquad=isquad):
    reward = 0
    y = len(setx[0])
    for i in range(y):
        if not isquad:  # Add check here
            # print("🛑 Squad empty - stopping auction")
            return [obs, score]
        active_player = rd.choice(setx[0])    # choose a player and use the csv to reference it and make an object to compare!
        newplyr = create_player(df=df,player=active_player)
        player = find_closest_player(newplyr=newplyr,isquad=isquad)
        relatability = relatability_score(newplyr=newplyr,player=player)
        x = setx[0].index(active_player)
        active = Player(name=active_player,role=setx[2][x],nationality=setx[1][x],price=setx[3])
        checker = ['no bid']*len(teams)  
        bid_action = agent.choose_action(obs)
        if bid_action > 0:
            while not active.isSold:
                budget = calculate_budget(relatibility=relatability, predicted_price=price_predictor(plyr=newplyr, df=pricesdf))
                bids = {}                                          # keeps bids inside itself ^ _ ^
                for i in range(len(teams)):
                    bids[teams[i]] = 'no bid'
                ai(player = active,bids = bids)                       # other 9 teams response
                bids_values = list(bids.values())
                if bids_values == checker:                         # if no one bids for the player, he's yours
                    adding(player=active,team = your_team)
                    removing(player=active,set_container=setx)
                    # print(f'{active.name} has been sold to {your_team.name} for {active.price}! ')
                    reward = calculate_reward(price=active.price,relatibility=relatability,budget=calculate_budget(relatibility=relatability, predicted_price=price_predictor(plyr=newplyr, df=pricesdf)))
                    new_state, reward, done, info = env.step(bid_action)
                    agent.remember(obs, bid_action, reward, new_state, int(done))
                    agent.learn()
                    score += reward
                    obs = new_state
                    del isquad[player.name]
                    if not isquad:  # Check after removal
                        # print("🛑 Final player sold - ending auction")
                        pass
                    return [obs, score]
                    active.isSold = True
                else:
                    if your_team.purse > active.price:
                        bid_action = agent.choose_action(obs)
                        if bid_action > 0:
                           continue
                        else:
                            for i in range(len(bids)-1,-1,-1):
                                if bids_values[i] == 'bid':
                                    bid_winner = list(bids.keys())[i]
                                    # print(f'{active.name} will be sold to {bid_winner} at {active.price}')
                                    reward = skip_reward(price=active.price,relatibility=relatability,budget=budget)
                                    new_state, reward, done, info = env.step(bid_action)
                                    agent.remember(obs, bid_action, reward, new_state, int(done))
                                    agent.learn()
                                    score += reward
                                    obs = new_state
                                    removing(player=active,set_container=setx)
                                    active.isSold = True
                    else:
                        for i in range(len(bids)-1,-1,-1):
                                if bids_values[i] == 'bid':
                                    bid_winner = list(bids.keys())[i]
                                    # print(f'{active.name} will be sold to {bid_winner} at {active.price}')       
                                    removing(player=active,set_container=setx)
                                    active.isSold = True
                                
        else:
            # print(f"{active.name} will remain unsold!, next player please!")
            reward = calculate_reward(price=active.price,relatibility=relatability,budget=calculate_budget(relatibility=relatability, predicted_price=price_predictor(plyr=newplyr, df=pricesdf)))
            new_state, reward, done, info = env.step(bid_action)
            agent.remember(obs, bid_action, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            removing(player=active,set_container=setx)
    # print(f'After the end of this set, this is how {your_team.name} looks like! \n {your_team.squad}')      
    return [obs, score]

sets = [marquee]    
score_history = []


for j in tqdm(range(num_iterations), desc="Training Progress", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"):
    obs = env.reset()
    score = 0
    isquad = create_squad(df=team_df)
    
    marquee_names, marquee_nationalities, marquee_roles = [], [], []

    with open('/Users/jatin/Documents/model stuff/auction2024list.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            full_name = re.sub(r'\s+', ' ', f"{row['First Name']} {row['Surname']}").strip()
            marquee_names.append(full_name.title())

            nationality = '🛺' if row['Country'] == 'India' else '✈️'
            marquee_nationalities.append(nationality)

            role = re.sub(r'\s+', ' ', row['Specialism']).strip().title()
            marquee_roles.append(role)

        marquee = [marquee_names, marquee_nationalities, marquee_roles, 20000000]
    
    sets = [marquee]
    for i in range(len(sets)):
        [obs, score] = bidding(setx=sets[i], agent=agent, env=env, obs=obs, score=score, isquad=isquad)

    score += loot_rewards(squad=your_team)  # Rewards based on squad structure
    score_history.append(score)

    if j % 25 == 0:
        agent.save_models()

    # print(f'Episode {j}, Score: {score:.2f}, Trailing 100 Avg: {np.mean(score_history[-100:]):.3f}')