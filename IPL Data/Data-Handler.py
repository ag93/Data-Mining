# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:44:15 2019

@author: Darshak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#warnings.simplfilter(action = "ignore", category = FutureWarning)

matches = pd.read_csv('matches.csv')
matches["type"] = "pre-qualifier"
for year in range(2008, 2017):
   final_match_index = matches[matches['season']==year][-1:].index.values[0]
   matches = matches.set_value(final_match_index, "type", "final")
   matches = matches.set_value(final_match_index-1, "type", "qualifier-2")
   matches = matches.set_value(final_match_index-2, "type", "eliminator")
   matches = matches.set_value(final_match_index-3, "type", "qualifier-1")

matches.groupby(["type"])["id"].count()
matches.head()

deliveries = pd.read_csv('deliveries.csv')
deliveries.head()
 
team_score = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().unstack().reset_index()
team_score.columns = ['match_id', 'Team1_score', 'Team2_score', 'Team1_superover_score', 'Team2_superover_score']
matches_agg = pd.merge(matches, team_score, left_on = 'id', right_on = 'match_id', how = 'outer')

team_extras = deliveries.groupby(['match_id', 'inning'])['extra_runs'].sum().unstack().reset_index()
team_extras.columns = ['match_id', 'Team1_extras', 'Team2_extras', 'Team1_superover_extras', 'Team2_superover_extras']
matches_agg = pd.merge(matches_agg, team_extras, on = 'match_id', how = 'outer')

#Reordering the data
cols = ['match_id', 'season','city','date','team1','team2', 'toss_winner', 'toss_decision', 'result', 'dl_applied', 'winner', 'Team1_score','Team2_score', 'win_by_runs', 'win_by_wickets', 'Team1_extras', 'Team2_extras', 'Team1_superover_score', 'Team2_superover_score', 'Team1_superover_extras', 'Team2_superover_extras', 'player_of_match', 'type', 'venue', 'umpire1', 'umpire2', 'umpire3']
matches_agg = matches_agg[cols]

# FOR BATSMEN

batsman_grp = deliveries.groupby(['match_id', 'inning', 'batting_team', 'batsman'])
batsmen = batsman_grp['batsman_runs'].sum().reset_index()

#Ignore the wide balls
balls_faced = deliveries[deliveries['wide_runs'] == 0]
balls_faced = balls_faced.groupby(['match_id', 'inning', 'batsman'])['batsman_runs'].count().reset_index()
balls_faced.columns = ['match_id', 'inning', 'batsman', 'balls_faced']
batsmen = pd.merge(balls_faced, batsmen, left_on=['match_id', 'inning', 'batsman'], 
                        right_on=['match_id', 'inning', 'batsman'], how='left')

Total_balls = balls_faced.groupby(['batsman'])['balls_faced'].sum().sort_values(ascending = False)
Total_balls = Total_balls.reset_index()

fours = deliveries[deliveries['batsman_runs'] == 4]
sixes = deliveries[deliveries['batsman_runs'] == 6]

fours_per_batsman = fours.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()
sixes_per_batsman = sixes.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

fours_per_batsman.columns = ["match_id", "inning", "batsman", "4s"]
sixes_per_batsman.columns = ["match_id", "inning", "batsman", "6s"]

batsmen = batsmen.merge(fours_per_batsman, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")
batsmen = batsmen.merge(sixes_per_batsman, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen['SR'] = np.round(batsmen['batsman_runs'] / batsmen['balls_faced'] * 100, 2)

for col in ["batsman_runs", "4s", "6s", "balls_faced", "SR"]:
    batsmen[col] = batsmen[col].fillna(0)
    
dismissals = deliveries[ pd.notnull(deliveries["player_dismissed"])]
dismissals = dismissals[["match_id", "inning", "player_dismissed", "dismissal_kind", "fielder"]]
dismissals.rename(columns={"player_dismissed": "batsman"}, inplace=True)
batsmen = batsmen.merge(dismissals, left_on=["match_id", "inning", "batsman"], 
                        right_on=["match_id", "inning", "batsman"], how="left")

batsmen = matches[['id','season']].merge(batsmen, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

batsmen['inning'] = batsmen['inning'].replace(to_replace = 2, value = 1)

# FOR BOWLERS


bowler_grp = deliveries.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])
bowlers = bowler_grp["total_runs", "wide_runs", "bye_runs", "legbye_runs", "noball_runs"].sum().reset_index()

bowlers["runs"] = bowlers["total_runs"] - (bowlers["bye_runs"] + bowlers["legbye_runs"])
bowlers["extras"] = bowlers["wide_runs"] + bowlers["noball_runs"]

del( bowlers["bye_runs"])
del( bowlers["legbye_runs"])
del( bowlers["total_runs"])

dismissal_kinds_for_bowler = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
dismissals = deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds_for_bowler)]
dismissals = dismissals.groupby(["match_id", "inning", "bowling_team", "bowler", "over"])["dismissal_kind"].count().reset_index()
dismissals.rename(columns={"dismissal_kind": "wickets"}, inplace=True)

bowlers = bowlers.merge(dismissals, left_on=["match_id", "inning", "bowling_team", "bowler", "over"], 
                        right_on=["match_id", "inning", "bowling_team", "bowler", "over"], how="left")
bowlers["wickets"] = bowlers["wickets"].fillna(0)

bowlers_over = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler'])['over'].count().reset_index()
bowlers = bowlers.groupby(['match_id', 'inning', 'bowling_team', 'bowler']).sum().reset_index().drop('over', 1)
bowlers = bowlers_over.merge(bowlers, on=["match_id", "inning", "bowling_team", "bowler"], how = 'left')
bowlers['Econ'] = np.round(bowlers['runs'] / bowlers['over'] , 2)
bowlers = matches[['id','season']].merge(bowlers, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)


bowlers['inning'] = bowlers['inning'].replace(to_replace = 2, value = 1)

#Combining data for batsmen

Total_runs = batsmen.groupby(['batsman'])['batsman_runs'].sum()
Total_runs = Total_runs.sort_values( ascending = False)
Total_runs = Total_runs.reset_index()

Total_wickets = bowlers.groupby(['bowler'])['wickets'].sum().reset_index()
#Total_wickets = Total_wickets.sort_values(ascending = False)
#print(Total_wickets)

Total_matches = batsmen.groupby(['batsman'])['inning'].sum().reset_index()
Total_matches = Total_matches.sort_values(by = 'inning', ascending = False)
Total_matches.rename(columns = {'inning':'Tot_matches'}, inplace = True)
print(Total_matches)


Total_Sr = pd.merge(Total_runs, Total_balls, left_on = 'batsman', right_on = 'batsman', how = 'left')
Total_Sr['Avg_Sr'] = (Total_Sr['batsman_runs']/Total_Sr['balls_faced'])*100
Total_Sr = Total_Sr.sort_values(by = 'Avg_Sr', ascending = False)

Total_4s = batsmen.groupby(['batsman'])['4s'].sum().reset_index()
Total_6s = batsmen.groupby(['batsman'])['6s'].sum().reset_index()

Total_NO = batsmen.groupby(['batsman'])['dismissal_kind'].count().reset_index()
Total_NO = Total_matches[['batsman','Tot_matches']].merge(Total_NO, left_on = 'batsman', right_on = 'batsman', how = 'left')
Total_NO['Total_NO'] = Total_NO['Tot_matches']-Total_NO['dismissal_kind']

#Combining data for bowlers

Total_Runs_Conceded = bowlers.groupby(['bowler'])['runs'].sum().reset_index()
Total_Overs_bowled = bowlers.groupby(['bowler'])['over'].sum().reset_index()
Total_Overs_bowled['Balls_bowled'] = Total_Overs_bowled['over']*6

Total_BigWicket = bowlers.loc[:,['bowler','wickets']]
Total_BigWicket = pd.crosstab(Total_BigWicket.bowler, Total_BigWicket.wickets).reset_index()
Total_BigWicket['Big_Wickets'] = Total_BigWicket[4]+Total_BigWicket[5]+Total_BigWicket[6]

Total_Bowler_innings = bowlers.groupby(['bowler'])['inning'].sum().reset_index()

# MAIN PARAMETERS FOR BATSMEN

Final_Batsmen_Parameters = pd.merge(Total_Sr, Total_runs, left_on = ['batsman','batsman_runs'], right_on = ['batsman','batsman_runs'], how = 'left')
Final_Batsmen_Parameters = Total_4s[['batsman','4s']].merge(Final_Batsmen_Parameters, left_on = 'batsman', right_on = 'batsman', how = 'left')
Final_Batsmen_Parameters = Total_6s[['batsman','6s']].merge(Final_Batsmen_Parameters, left_on = 'batsman', right_on = 'batsman', how = 'left')
Final_Batsmen_Parameters = Total_NO[['batsman', 'Tot_matches','Total_NO','dismissal_kind']].merge(Final_Batsmen_Parameters, left_on = 'batsman', right_on = 'batsman', how = 'left')

Final_Batsmen_Parameters['Hard_Hitter'] = (Final_Batsmen_Parameters['4s']*Final_Batsmen_Parameters['6s'])/Final_Batsmen_Parameters['balls_faced']
Final_Batsmen_Parameters['Finisher'] = Final_Batsmen_Parameters['Total_NO']/Final_Batsmen_Parameters['Tot_matches']
Final_Batsmen_Parameters['Fast_Scorer'] = Final_Batsmen_Parameters['batsman_runs']/Final_Batsmen_Parameters['balls_faced']
Final_Batsmen_Parameters['Consistent'] = Final_Batsmen_Parameters['batsman_runs']/Final_Batsmen_Parameters['dismissal_kind']
Final_Batsmen_Parameters['Running'] = (Final_Batsmen_Parameters['batsman_runs']-(Final_Batsmen_Parameters['4s']+Final_Batsmen_Parameters['6s']))/(Final_Batsmen_Parameters['balls_faced']-(Final_Batsmen_Parameters['4s']+Final_Batsmen_Parameters['6s']))

# MAIN PARAMETERS FOR BOWLERS

Final_Bowler_Parameters = pd.merge(Total_Runs_Conceded, Total_Overs_bowled, left_on = ['bowler'], right_on = ['bowler'], how = 'left')
Final_Bowler_Parameters = Total_wickets.merge(Final_Bowler_Parameters, left_on = 'bowler', right_on = 'bowler', how = 'left')
Final_Bowler_Parameters = Total_BigWicket[['bowler', 4, 5, 6,'Big_Wickets']].merge(Final_Bowler_Parameters, left_on = 'bowler', right_on = 'bowler', how = 'left')
Final_Bowler_Parameters = Total_Bowler_innings[['bowler','inning']].merge(Final_Bowler_Parameters, left_on = 'bowler', right_on = 'bowler', how = 'left')

Final_Bowler_Parameters['Economy'] = Final_Bowler_Parameters['runs']/Final_Bowler_Parameters['over']
Final_Bowler_Parameters['Wicket_Taker'] = Final_Bowler_Parameters['Balls_bowled']/Final_Bowler_Parameters['wickets']
Final_Bowler_Parameters['Big_Wicket_Taker'] = Final_Bowler_Parameters['Big_Wickets']/Final_Bowler_Parameters['inning']
Final_Bowler_Parameters['consistent'] = Final_Bowler_Parameters['runs']/Final_Bowler_Parameters['wickets']
Final_Bowler_Parameters['Short_Performance'] = (Final_Bowler_Parameters['wickets']-4*Final_Bowler_Parameters[4]-5*Final_Bowler_Parameters[5]-6*Final_Bowler_Parameters[6])/(Final_Bowler_Parameters['inning']-Final_Bowler_Parameters['Big_Wickets'])


