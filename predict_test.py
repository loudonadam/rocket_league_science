#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

category = {
    'is_ot': 0, 
    'game_time': 389, 
    'boost_per_min': 405, 
    'boost_avg_amt': 48,
    'big_stoln_per_min': 1.4, 
    'big_clctd_per_min': 3.2, 
    'sml_clctd_per_min': 12.8,
    'pct_zero_boost': 3, 
    'pct_full_boost': 7, 
    'pct_0_25_boost': 31, 
    'pct_25_50_boost': 19,
    'pct_75_100_boost': 10, 
    'avg_speed': 1322,  
    'pct_supersonic': 21, 
    'pct_slow': 50,
    'avg_powerslide_duration': .22, 
    'powerslide_per_min': 9.3, 
    'pct_ground': 63,
    'pct_high_air': 5, 
    'avg_dist_to_ball': 2001, 
    'avg_dist_to_mates': 3022,
    'pct_def_third': 32, 
    'pct_off_third': 23, 
    'pct_behind_ball': 88, 
    'pct_most_back': 49,
    'percent_closest_to_ball': 17, 
    'demos_given_per_min': 1.6, 
    'demos_taken_per_min': .4,
    'tm8_G': 0, 
    'tm8_J': 1, 
    'tm_color_blue': 0, 
    'tm_color_orange': 1, 
    'car_': 0,
    'car_BMW 1 Series': 0, 
    'car_Battle Bus': 0, 
    'car_Fennec': 1,
    'car_Nissan Fairlady Z': 0, 
    'car_Nissan Silvia': 0, 
    'car_Octane': 0,
    'car_Porsche 911 Turbo': 0, 
    'car_Porsche 911 Turbo RLE': 0, 
    'car_Vulcan': 0
}

response = requests.post(url,json=category).json()

print (response)

if response['game_win_guess']==True:
    print('Considering your game data, I would guess you would WIN this game.')
else:
    print('Considering your game data, I would guess you would LOSE this game.')
# In[ ]:




