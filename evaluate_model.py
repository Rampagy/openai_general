import gym
import NeuralNetwork as nn
import numpy as np


def EvalModel(model, env=gym.make('Acrobot-v1')):
    cumulative_reward = 0
    num_of_games = 10

    for i in range(num_of_games):
        done = False
        obs_img = np.zeros((1, 10, 6, 1))

        observation = env.reset()

        while not done:
            if i < 1:
                # render for viewing experience
                env.render()

            obs_img = np.roll(obs_img, 1, axis=1)
            obs_img[0, 0, :, 0] = observation

            action = model.predict_move(obs_img, train=False)

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        print('current average: {} in {} games'.format(cumulative_reward/(i+1), (i+1)))

    print('average score: {}'.format(cumulative_reward/num_of_games))
    return cumulative_reward/num_of_games



if __name__ == "__main__":
    # create model
    model = nn.Control_Model()
    EvalModel(model)
