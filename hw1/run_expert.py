#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import os

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    loader = tf.train.import_meta_graph('humanoid_nn/model.meta')
    pred = tf.get_default_graph().get_tensor_by_name("pred:0")

    with tf.Session() as sess:
        tf_util.initialize()
        loader.restore(sess, os.path.dirname(os.path.abspath(__file__)) + "/humanoid_nn/model")

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(pred, feed_dict={'X:0': obs[None,:], 'keep_prob:0': 1.0})
                observations.append(obs)
                # Append the expert action.
                actions.append(policy_fn(obs[None,:]))
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                      'actions': np.squeeze(np.array(actions))}
        print "#observations ", np.array(observations).shape, ", #actions ", np.array(actions).shape
        pickle.dump(expert_data, open('data.pkl', 'wb'))

if __name__ == '__main__':
    main()
