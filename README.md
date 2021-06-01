# MPG_experiments

The files in this folder generate the plots for the experimental section of the paper (main part and supplementary material).

Main File:

1. congestionMPG_plots.py: this code implement the policy gradient algorithm. For each policy profile, it has subroutines that estimate the discounted visitation distribution, the 
value function (of each agent) and the Q-function (of each agent). The stopping criterion is that two successive updates in the policy space do not differ more than a constant
in the L1-norm. The constant that we used in the experiments is 10e-16.

Auxiliary Files:

1. projection_simplex.py: this code does the projection to the simplex in the projected gradient ascent (policy gradient) updates.
This file is due to: https://gist.github.com/EdwardRaff/f4f4cf0c927c2addfb39

2. congestion_games.py: this code creates the class "Congestion Game". Each Congestion Game comprises a set of players, a set of facilities (that the players can 
choose from) and a set of weights (one for each facility). The methods of the Class and the additional functions mainly aim to calculate the agents' rewards (which
in these instances are equal to the number of players at each facility times the weight of that facility).

Other Files:

The files congestionMPG_lrs (lrs = learning rates) and congestionMPG_ord (ord = ordinal MPG) implement the variations of the main experiment 
(experiment of Section 5 of the main paper) that are shown in the supplementary material. Their structure is very similar to the main file.
