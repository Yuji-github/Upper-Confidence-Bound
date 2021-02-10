# Upper Confidence Bandit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def ucb():
    # import dataset
    dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
    # print(dataset)
    '''
          Ad 1  Ad 2  Ad 3  Ad 4  Ad 5  Ad 6  Ad 7  Ad 8  Ad 9  Ad 10
    0        1     0     0     0     1     0     0     0     1      0
    1        0     0     0     0     0     0     0     0     1      0
    9999     0     1     0     0     0     0     0     0     0      0
    [10000 rows x 10 columns]
    this is for a simulation, the dataset is recorded before. 
    no updating in real time and we know which ads are selected
    '''

    # implementation for UCB

    # before steps
    N = 10000 #(10000 rows)
    d = 10  # (10 columns)
    ads_selected = [] # lists for selected

    # step 1
    num_of_selection = [0]*d # the num of times ad i was selected up to round n: once it is selected, the value is increasing by 1; ex) a user selects ad 2 then, ad2 = [1]
    sum_of_reward = [0]*d # the SUM of rewards ad i in round n
    total_reward = 0 # to store total reward and start is 0 because it is not selected yet

    # step 2-1: the average reward of ad i up to round n
    for n_round in range(0, N):
        ad = 0 # traverse the columns
        max_upper_bound = 0 # check the highest upper bound and starts is 0 as no selected

        for i in range(0, d): # traverse in the columns
            if (num_of_selection[i] > 0):
                average_reward = sum_of_reward[i] / num_of_selection[i]

                # step 2-2: calculate the confidence interval
                confidence_interval = math.sqrt(3/2 * math.log(n_round+1) / num_of_selection[i]) # n_round+1 => avoiding 0

                # step 3-1: create maximum UBC
                upper_bound = average_reward + confidence_interval

            else: # step 3-1: create maximum UBC
                upper_bound = 1e400 # if they are not selected: we can set the upper bound super high

            if (max_upper_bound < upper_bound): # step 3-2: select maximum UBC
                max_upper_bound = upper_bound
                ad = i # update

        ads_selected.append(ad)
        num_of_selection[ad] = num_of_selection[ad] + 1
        reward = dataset.values[n_round, ad]
        sum_of_reward[ad] = sum_of_reward[ad] + reward
        total_reward = total_reward + reward

    # visualizing data
    plt.hist(ads_selected)
    plt.title('Selection of Ads')
    plt.xlabel('Ads')
    plt.ylabel('Rewards')
    plt.show()

if __name__ == '__main__':
    ucb()