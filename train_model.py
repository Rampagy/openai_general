import gym
import NeuralNetwork as nn
import evaluate_model as em
import numpy as np


# create model
model = nn.Control_Model()
env = gym.make('Acrobot-v1')
train_count = 0

# train to 500 until the average is above 500
while(em.EvalModel(model, env) < -100):
    max_reward = -600

    # run a segment of 200 'games' and train off of the max score
    for i in range(200):
        cumulative_reward = 0
        obs_log = []
        action_log = []
        done = False
        obs_img = np.zeros((10, 6))

        observation = env.reset()

        while not done:
            obs_img = np.roll(obs_img, 1, axis=0)
            obs_img[0, :] = observation

            reshaped = obs_img.reshape((1, 60))

            action = model.predict_move(reshaped)

            # keep a log of actions and observations
            obs_log += [reshaped.flatten()]
            action_log += [action]

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        if cumulative_reward > max_reward:
            max_reward = cumulative_reward
            max_obs_log = obs_log
            max_action_log = action_log

        print('Episode {} scored {}, max {}'.format(i, cumulative_reward, max_reward))

        if max_reward >= 500:
            # if perfect score has already been acheived
            break

    if max_reward > -500:
        train_count += 1
        # train the dnn
        model.train_game(max_obs_log, max_action_log)


print('{} training episodes'.format(train_count))
# save the model for evaluation
model.save_model()
