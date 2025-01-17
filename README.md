# Rocket League Science
![image](https://github.com/user-attachments/assets/d697ff79-54af-4fd5-815d-ec470cdbc1cd)


## Overview
The video game I have played the most in my life is Rocket League. It's an amazingly simple game which pitches two teams against eachother in a game of soccer. In this case, however, the players are cars which can fly. This game has almost no overlapping skills with any other video game and therefore can have a very steep learning curve for new players. 

For years, I have wanted to analyze my Rocket League data in an attempt to see what strategies and details are the most important to winning a game. There is an amazing community tool called ballchasing.com which allows you to upload your game data after each game to their site. Then I can access this data either using their visualisations and tables or using their API. The data that is uploaded is VERY specific. In fact, you could use the uploaded data to reconstruct a complete replay of the game. Also, the site calculates certain metrics using that data. These are the metrics which I will use here. Without getting into the weeds of Rocket League, the data can include how many times you demolished an opponent, what percentage of the time were you in the air, what percentage of the time were you in the offensive third, and so on. The data points I have chosen to use are those which I think are helpful to predict victory, but not directly responsible. For example, how many goals I score would obviously be highly indicative of a win, because you can't win without scoring. I am more interested in how positional, strategy, and speed metrics affect the outcome so I will limit myself to those indirect indicators.

## Data Collection
Frankly, the most difficult part of this project was the data collection. I needed to learn how to use ballchasing.com's API, digest the json files it gave me, clean it up, and then prepare usable data. In [this](https://github.com/loudonadam/rocket_league_science/blob/main/Data/Replay%20ID%20Fetch.ipynb) script, I fetch all of the replays IDs uploaded to the site where I am the player and the game mode is 2v2. I store that list in this [file](https://github.com/loudonadam/rocket_league_science/blob/main/Data/replay_ids) and then use a second [script](https://github.com/loudonadam/rocket_league_science/blob/main/Data/RL%20Data%20Dec22.ipynb) to loop through all of those IDs, fetch the individual game data, and produce the desired metrics. The [final data](https://github.com/loudonadam/rocket_league_science/blob/main/Data/replay_data_2.csv) is a table where each row represents a single game marked as a win or loss and containing ~40 variables I can use to predict the outcome.


## General Data Analysis
Because I prepared the data myself, I've already cleaned it quite thoroughly. However, it it important that I understand if I am including any variable that are too impactful to the model. For example, I found that the metric **% of time in offensive third** was the most highly correlated variable to victory. This makes since, because if I am scoring a lot and putting on a lot of pressure, it would make sense that I would be on offense frequently. I ultimately decided to keep this data point, but in the future it could be interesting to see how results might look if I were to ignore it.

For the most part, the variables I have seem to all have low correlations with winning. For this analysis, I will use all of them. In a future project I may attempt to be more selective. I'm working on a script to scrape player data from a different website that would tell me how many games total an opponent has played, however that code is still incomplete, so is not included here. For now, the project has resulted in a model with mild predictive success for Rocket League games, but I'd like to tease out additional improvements over time.
