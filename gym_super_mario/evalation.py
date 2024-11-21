import numpy as np
import torch
from core.model.wrappers import  wrap_environment
from core.model.model import CNNDQN
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,
                                          RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
model_path = ''

if __name__ == '__main__':
    flag = False
    # Initialize our environment. Specify the level and strategy here.
    env = wrap_environment("SuperMarioBros-1-1-v0", SIMPLE_MOVEMENT)
    # Initialize our neural network with input information and the number of actions we can perform.
    # (4, 84, 84) 7
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    # Manually load our model.
    net.load_state_dict(torch.load("pretrained_models/SuperMarioBros-1-1-v0.dat"))
    # Initialize reward information and reset our environment.
    total_reward = 0.0
    state = env.reset()
    while True:
        # The 'state' here is the current state of our game simulator, which is an array of shape (4, 84, 84).
        state_v = torch.tensor(np.array([state], copy=False))
        # We use our network to make predictions and get the next action.
        # This returns a list of predictions for each of the 7 possible actions.
        # For example: [79.629974, 81.05619, 80.52793, 83.71866, 71.73175, 79.65819, 81.28025]
        q_vals = net(state_v).data.numpy()[0]
        # Then, we select the action with the highest value from this model. It will return a value from 0 to 6.
        action = np.argmax(q_vals)
        # We provide the environment with the action we intend to execute, and the environment returns 4 parameters:
        # - 'reward' represents the reward obtained in this step.
        # - 'done' indicates whether the game has ended.
        # - 'info' contains some game information, such as: {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 392, 'world': 1, 'x_pos': 397, 'x_pos_screen': 112, 'y_pos': 181}
        state, reward, done, info = env.step(action)
        # Rendering the interface is mainly for observing the state; this part is optional.
        env.render()
        # Here, we can get the current total reward. The ultimate goal of our game is to maximize this reward and reach the flag.
        total_reward += reward
        # If we capture the flag, it means we have succeeded.
        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            print(info)
            flag = True
        if done:
            # After completing the game, print the total reward.
            print(total_reward)
            break

    env.close()
