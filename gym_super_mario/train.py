from core.model.replay_buffer import PrioritizedBuffer
from core.model.train_information import TrainInformation
from core.model.wrappers import wrap_environment
from os.path import join
from shutil import copyfile, move
from core.model.test import test
from core.model.helpers import (compute_td_loss,
                                initialize_models,
                                set_device,
                                update_beta,
                                update_epsilon)
from torch import save
from torch.optim import Adam
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT,
                                          RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)


# Update our model
def update_graph(model, target_model, optimizer, replay_buffer, device, info, beta):
    if len(replay_buffer) > initial_learning:
        if not info.index % target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        # Calculate Q information
        compute_td_loss(model, target_model, replay_buffer, gamma, device, batch_size, beta)
        optimizer.step()

def test_new_model(model, info):
    # Save our model
    save(model.state_dict(), join('pretrained_models', '%s.dat' % environment))
    print('Testing model...')
    # Test our model
    flag = test(environment, action_space, info.new_best_counter)
    # If the model passes the test, we copy it
    if flag:
        copyfile(join('pretrained_models', '%s.dat' % environment), 'recording/run%s/%s.dat' % (info.new_best_counter, environment))

# Complete this round of training
def complete_episode(model, info, episode_reward, episode, epsilon, stats):
    # First, calculate the reward score for this round of training
    new_best = info.update_rewards(episode_reward)
    # If it's a new record, let's test our model
    if new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
        # Here, we pass the model
        test_new_model(model, info)
    elif stats['flag_get']:
        # If we got the flag in this training episode, record the best training count
        info.update_best_counter()
        test_new_model(model, info)
    print('Episode %s - Reward: %s, Best: %s, Average: %s Epsilon: %s' % (episode,
                                                                           round(episode_reward, 3),
                                                                           round(info.best_reward, 3),
                                                                           round(info.average, 3),
                                                                           round(epsilon, 4)))


# Start one training episode
def run_episode(env, model, target_model, optimizer, replay_buffer,
                device, info, episode):
    # Set the current training total score and reset the environment
    episode_reward = 0.0
    state = env.reset()

    while True:
        # Calculate
        epsilon = update_epsilon(info.index)
        if len(replay_buffer) > batch_size:
            beta = update_beta(info.index)
        else:
            beta = 0.4
        # Call our model to get the next action
        action = model.act(state, epsilon, device)
        # Whether to display the game interface
        if render:
            env.render()
        # Pass our model's prediction results to the simulator
        next_state, reward, done, stats = env.step(action)
        # Put the data we explored into the experience replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        # Get the next state and update the score
        state = next_state
        episode_reward += reward
        # Update the current index, which is essentially +1 operation
        info.update_index()
        # Update the weight information of our model
        update_graph(model, target_model, optimizer, replay_buffer,
                     device, info, beta)
        # If the game is over, we complete this round of training
        if done:
            complete_episode(model, info, episode_reward,
                             episode, epsilon, stats)
            break


# Model training
def train(env, model, target_model, optimizer, replay_buffer, device):
    # Information about the current training
    info = TrainInformation()

    for episode in range(num_episodes):
        # Start each round of training
        run_episode(env, model, target_model, optimizer, replay_buffer,
                    device, info, episode)


# Main function
def main():
    # Initialize the environment
    env = wrap_environment(environment, action_space)
    # Set our device (GPU or CPU)
    device = set_device(force_cpu)
    # Initialize our models, including two models to facilitate fitting
    model, target_model = initialize_models(environment, env, device, transfer)
    # Set our optimizer for saving states and updating parameters
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # Set up the replay buffer
    replay_buffer = PrioritizedBuffer(buffer_capacity)
    # Start training our model
    train(env, model, target_model, optimizer, replay_buffer, device)
    # Close the environment after training
    env.close()

# Training parameter configuration
# Level configuration
environment = "SuperMarioBros-1-1-v0"
# Action space mode
action_space = SIMPLE_MOVEMENT
# Whether to force the use of CPU for computation
force_cpu = False
# Whether to use pre-trained model weights (can speed up training)
transfer = True
# Learning rate setting
learning_rate = 1e-4
# Size of the experience replay buffer
buffer_capacity = 20000
# Number of training episodes
num_episodes = 50000
# Number of data samples to take at once
batch_size = 32
# Whether to display the game interface
render = True
# After how many attempts to start updating the model officially
initial_learning = 10000
# Model update frequency
target_update_frequency = 1000
# Discount factor for future rewards
gamma = 0.99

if __name__ == '__main__':
    main()
