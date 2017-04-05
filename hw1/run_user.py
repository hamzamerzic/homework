#!/usr/bin/env python

import pickle
import numpy as np
import gym
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def main():
    data = pickle.load(open('data-ant.pkl', 'rb'))
    observations = data['observations']
    actions = data['actions']
    actions = np.reshape(actions, (len(actions), len(actions[0][0])))

    X_train, X_test, y_train, y_test = train_test_split(
        observations, actions, test_size=0.33)

    #reg = Ridge(alpha=1)
    reg = RandomForestRegressor(max_depth=30, min_samples_leaf=5, n_jobs=-1)
    reg.fit(X_train, y_train)
    joblib.dump(reg, 'model_ant-rf.pkl')

    #reg = joblib.load('model_ant.pkl')
    print reg.score(X_test, y_test)


if __name__ == '__main__':
    main()

# Ridge:   4262.0296482026806
# RF:      4619.1549077987811
# Expert:  4828.1993194265915
