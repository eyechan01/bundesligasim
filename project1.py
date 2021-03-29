import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the 2019-2020 Bundesliga data
data = pd.read_csv("/Users/EthanLee/Desktop/STAT 143/bundes1819.csv")

# Home and away goals
homegoals = data.FTHG.values
awaygoals = data.FTAG.values

# Get unique teams, number of teams, and number of games
teams = np.unique(data[['HomeTeam']])
nteams = teams.size
ngames = data.HomeTeam.size

# Create design matrix
game_id = np.arange(ngames)
Xh = np.zeros((ngames, nteams))
home_team_idx = np.searchsorted(teams,data.HomeTeam)
away_team_idx = np.searchsorted(teams,data.AwayTeam)
for i in range(ngames):
    Xh[game_id[i],home_team_idx[i]] = 1
    Xh[game_id[i],away_team_idx[i]] = -1
X = np.vstack((Xh,-Xh))

# Create W matrix
W = np.eye(nteams)
W[-1,:] = -1
W = W[:,:-1]

# Get the design matrix fulfilling constraint
Xs = X@W
Xs = np.c_[Xs,np.concatenate((np.ones(ngames),-1*np.ones(ngames)))]

k = np.concatenate((homegoals,awaygoals))
Xs = sm.add_constant(Xs)

# Set up and fit poisson model
poisson_model = sm.Poisson(k, Xs)
poisson_res = poisson_model.fit()
print(poisson_res.summary())

# Get parameters
intercept = poisson_res.params[0]
beta = poisson_res.params[-1]
theta_att = poisson_res.params[1:nteams]

# Get thetas for the final team (found using linear constraint)
theta_att = np.append(theta_att,-np.sum(theta_att))
# Use thetas to get expected goals per game
lambda_att = np.exp(intercept + theta_att)

# Print out results
team_params = []
for t,a,l in zip(teams,lambda_att,theta_att):
    team_params.append((t,a.round(3),l.round(3)))

team_params = sorted(team_params,key=lambda x: x[1], reverse=False)
for params in team_params[::-1]:
    print("%s,%1.3f,%1.3f" % tuple(params))
print("Beta,%1.3f" % (beta))

# Define dict containing team strength parameters
team_param_dict = {}
for team in team_params:
    team_param_dict[team[0]] = team[2]

def draw_from_strength_params(teams,team_param_dict,sigma_teams=0.1,sigma_beta=0.3):
    '''
    Draw a new set of team strength, home advantage and intercept parameters from the standard errors on the theta values
    '''
    sim_param_dict = team_param_dict.copy()
    for team in teams:
        sim_param_dict[team] = stats.norm.rvs(0,sigma_teams,1)[0] + team_param_dict[team]

    sim_param_dict['intercept'] = stats.norm.rvs(0,sigma_beta,1)[0] + intercept
    sim_param_dict['beta'] = stats.norm.rvs(0,sigma_beta,1)[0] + beta
    return sim_param_dict

def sim_match(home,away,team_param_dict):
    # Calculate lambda values for Poission distribution using team strength and home advantage
    lambda_home = np.exp( team_param_dict[home] - team_param_dict[away] + intercept + beta )
    lambda_away = np.exp( team_param_dict[away] - team_param_dict[home] + intercept - beta )
    # Draw a poisson random variable
    hg = stats.poisson.rvs(lambda_home)
    ag = stats.poisson.rvs(lambda_away)
    return hg,ag

def sim_season(fixtures,team_param_dict):
    '''
    simulate a single season using the fixtures given in the 'fixtures' dataframe
    '''
    # Make copy of fixutre list that starts with zeros
    completed_fixtures = fixtures.copy()
    nmatches = fixtures.shape[0]
    HomeGoals = np.zeros(nmatches,dtype=int)
    AwayGoals = np.zeros(nmatches,dtype=int)
    # Simulate each match in fixture schedule
    for i,row in completed_fixtures.iterrows():
        HomeGoals[i],AwayGoals[i] =sim_match(row['HomeTeam'],row['AwayTeam'],team_param_dict)
    # Fill in scores in fixture schedule
    completed_fixtures['HomeGoals'] = HomeGoals
    completed_fixtures['AwayGoals'] = AwayGoals
    return completed_fixtures

def generate_table(results):
    '''
    produce a league table from a simulation of a single team
    '''
    teams = np.unique( results[['HomeTeam','AwayTeam']] )
    table = []
    # Loop through each team
    for team in teams:
        # Get all the team's home & away games
        homegames = results[results['HomeTeam']==team]
        awaygames = results[results['AwayTeam']==team]
        # Calculate total number of goals scored
        GoalsFor = np.sum(homegames.HomeGoals) + np.sum(awaygames.AwayGoals)
        # Calculate total number of goals conceded
        GoalsAgainst = np.sum(awaygames.HomeGoals) + np.sum(homegames.AwayGoals)
        # Calculate total number of points won in home games
        HomePoints = 3*np.sum(homegames.HomeGoals>homegames.AwayGoals) + 1*np.sum(homegames.AwayGoals==homegames.HomeGoals)
        # Calculate total number of points won in away games
        AwayPoints = 3*np.sum(awaygames.AwayGoals>awaygames.HomeGoals) + 1*np.sum(awaygames.AwayGoals==awaygames.HomeGoals)
        # Add to get total points
        TotalPoints = HomePoints + AwayPoints
        # Append results to the table list
        table.append((team,TotalPoints,GoalsFor-GoalsAgainst,GoalsFor,GoalsAgainst))
    # Sort the table by points, then goal difference, then goals for
    table = sorted(table, key = lambda x: (x[1],x[2],x[3]), reverse=True)
    return table

def analyse_sim_results(tables,teams,printout=True):
    '''
    analyse the simulation results over all simulations
    '''
    team_summary = []
    nsims = len(tables)
    for team in teams:
        # Final positions of the team in each simulation
        positions = np.array( [[t[0] for t in table].index(team)+1 for table in tables] )
        # Final number of points of the team in each simulation
        points = np.array([table[position-1][1] for table,position in zip(tables,positions)])

        # Generate bar plot of percent chance of team finishing in each position
        positionPercentages = []
        for i in range(1, 19):
            positionPercentages.append(np.sum(positions==i)/nsims*100)
        xVals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(xVals, positionPercentages, 1)
        plt.xlabel("League Position")
        plt.ylabel("Percent Chance")
        plt.title(team + " League Position Chances")
        plt.xticks(np.arange(1, 19, 1))
        plt.show()

        # Proportion of sims in which team finished first in the league table
        title = np.sum(positions==1)/nsims*100
        # Proportion of sims in which team finished in the top 4
        top4 = np.sum(positions<=4)/nsims*100
        # Proportion of sims in which team finished in the bottom 3
        rel = np.sum(positions>=17)/nsims*100
        # Average position
        position_mean = np.mean(positions)
        # 2.5th percentile of finishes (by position)
        position_low = np.percentile(positions,2.5)
        # 97.5th percentile of finishes
        position_high = np.percentile(positions,97.5)
        # Average points
        points_mean = np.mean(points)
        # 2.5th and 97.5th percentiles of points
        points_low = np.percentile(points,2.5)
        points_high = np.percentile(points,97.5)
        # Add to summary table
        team_summary.append( [position_mean,team,points_mean,points_low,points_high,position_low,position_high,title,top4,rel] )
    # Sort summary table by mean position
    team_summary = sorted( team_summary, key = lambda x: x[0], reverse=False)
    if printout:
        print("\n******* SUMMARY OF SIMULATION RESULTS **********\n")
        print("Position, Team,Points-Mean,Points-Low,Points-High,Position-Low,Position-High,Title,Top4,Relegation")
        for i,summary in enumerate(team_summary):
            print("%d,%s,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f" % tuple([i+1]+summary[1:]))
    return team_summary

def run_sims(team_param_dict,nsims=10000,draw_strengths=True):
    # Get fixture schedule
    fixtures = pd.read_csv('/Users/EthanLee/Desktop/STAT 143/bundes1920.csv')
    fixtures = fixtures[['HomeTeam','AwayTeam']]
    teams = np.unique( fixtures[['HomeTeam','AwayTeam']] )

    # Calculate the average of team strength of teams that were relegated in 2018/19
    relegated_teams_theta = (team_param_dict['Nurnberg']+team_param_dict['Stuttgart'] + team_param_dict['Hannover'])/3.
    # Assign this average strength to the newly promoted teams
    promoted_teams = ['FC Koln','Paderborn', 'Union Berlin']
    for promoted_team in promoted_teams:
        team_param_dict[promoted_team] = relegated_teams_theta

    # Store the league tables for each simulation in a list
    tables = []
    # Run the simulations
    for i in tqdm(np.arange(nsims)):
        if draw_strengths:
            # Draw team strength from distribution determined by team's expected value of team strength and sample error
            sim_param_dict = draw_from_strength_params(teams,team_param_dict)
        else:
            # Use fixed expected values of team strengths
            sim_param_dict = team_param_dict
        # Simulate the season once
        completed_fixtures = sim_season(fixtures,sim_param_dict)
        # Append the league table for simulated season to the list
        tables.append( generate_table(completed_fixtures) )

    team_summary = analyse_sim_results(tables, teams)
    return tables,team_summary

run_sims(team_param_dict)
