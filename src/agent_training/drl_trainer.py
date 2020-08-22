from game.training_environment import TrainingEnv
from static.settings import *
from double_dqn.agent import DQNAgent
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, ReLU, Dropout, Flatten
from matplotlib import pyplot as plt
import json


# def build_cnn_model(input_shape, output_shape) -> Model:
#     n_actions = output_shape[0]
#     input_tensor = Input(shape=input_shape)
#     conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
#     max = MaxPool2D(padding='same')(conv)
#     relu = ReLU()(max)
#     for i in range(8):
#         conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(relu)
#         relu = ReLU()(conv)
#         dropout = Dropout(0.3)(relu)
#     flat = Flatten()(dropout)
#     fc = Dense(100, activation='relu')(flat)
#     fc = Dense(100, activation='relu')(fc)
#     final = Dense(n_actions)(fc)
#     model = Model(inputs=input_tensor, outputs=final)
#     model.compile(optimizer='adam', loss='mse')
#     return model

def build_small_fc_model(input_shape, output_shape, num_layers=3) -> Model:
    n_actions = output_shape[0]
    input_tensor = Input(shape=input_shape)
    fc = Dense(60, activation='relu')(input_tensor)
    for i in range(num_layers):
        drop = Dropout(0.5)(fc)
        fc = Dense(40, activation='relu')(drop)
    final = Dense(n_actions)(fc)
    fc_model = Model(inputs=input_tensor, outputs=final)
    fc_model.compile(optimizer='adam', loss='mse')
    print(fc_model.summary())
    return fc_model


def train_agent(architecture_path, weight_path, training_data_path, with_positions=True):
    game = TrainingEnv(['r'], training_mode=True, with_positions=with_positions)
    agent = DQNAgent(game)
    agent.set_model(build_small_fc_model(agent.get_state_shape(), agent.get_action_shape()))
    with open(architecture_path, 'w') as json_file:
        config = agent.to_json()
        json.dump(config, json_file)
    rewards, num_actions = agent.train(10000, weight_path, training_data_path, None, 128)

    # # Initiate data structures
    # training_sessions = 50
    # batch_size = 512
    # game = TrainingEnv(['r', 'r'], training_mode=True)
    # agent = DQNAgent(game)
    # agent.set_model(build_cnn_model(agent.get_state_shape(), agent.get_action_shape()))
    # config = agent.to_json()
    # accum_rewards = pd.DataFrame()
    #
    # # save model architecture
    # with open(os.path.join(model_path, 'model_architecture'), 'w') as json_file:
    #     json.dump(config, json_file)
    #
    # for i in tqdm(range(training_sessions)):
    #     if i == 1:
    #         # Set player to be reinforcement player
    #         with open(os.path.join(model_path, 'model_architecture')) as json_file:
    #             model = model_from_json(json.load(json_file))
    #             game.set_player(1, model)
    #     if i >= 1:
    #         # Set players weights to be the trained weights from last session
    #         game.players[1]._net.load_weights(model_path + f'/session_{i}_weights')
    #
    #     rewards, num_actions = agent.train(2, None, batch_size)
    #
    #     # Save all values that need to be saved
    #     accum_rewards = accum_rewards.append(rewards)
    #     agent.save_weights(os.path.join(model_path, f'session_{i + 1}_weights'))
    #     with open(os.path.join(model_path, 'rewards.csv'), 'w') as reward_file:
    #         accum_rewards.to_csv(reward_file)
    #
    # # If we finished all training sessions, we plot the rewards for each episode played
    # plt.scatter(range(1, len(accum_rewards) + 1), accum_rewards)
    # plt.title('Reward achieved over training episodes')
    # plt.xlabel('episode')
    # plt.ylabel('reward')
    # plt.savefig(os.path.join(model_path, f'reward_plot'))
    # plt.show()
    # model_path = r"/content/drive/My Drive/AchtungDieKurve/models/colab_tryout_model_box"

    training_sessions = 1
    game = TrainingEnv(['r'], training_mode=True)
    agent = DQNAgent(game)
    agent.set_model(build_fc_model(agent.get_state_shape(), agent.get_action_shape()))
    weight_path = model_path + os.path.sep + 'weights'
    checkpoint_path = model_path + os.path.sep + 'checkpoints'
    with open(model_path + os.path.sep + 'model_architecture.txt', 'w') as json_file:
        config = agent.to_json()
        json.dump(config, json_file)
    rewards, num_actions = agent.train(20, weight_path, checkpoint_path, None, 128)
    # for i in tqdm(range(training_sessions)):
    # if i == 1:
    # with open(model_path + '/model_architecture.txt', 'r') as json_file:
    #     config = json.load(json_file)
    #     model = model_from_json(config)
    # game.set_player(1, model_path + '/model_architecture.txt', model_path + '/session_1_weights')
    # if i >= 2:
    # game.players[1]._net.load_weights(model_path + f'/session_{i}_weights').expect_partial()
    # with open(model_path + '/exploration_rate') as exploration:
    #     exp = exploration.readlines()
    #     agent.exploration_rate = float(exp[0])
    # rewards, num_actions = agent.train(500, None, 512)
    # print('average rewards = ', np.average(rewards))
    # accum_rewards = accum_rewards.append(rewards)
    # actions_per_game = actions_per_game.append(num_actions)
    # agent.save_weights(model_path + f'/session_{i + 1}_weigfinal message = hts')
    # with open(model_path + '/rewards.csv', 'w') as reward_file:
    #     accum_rewards.to_csv(reward_file)
    # with open(model_path + '/num_actions.csv', 'w') as reward_file:
    #     accum_rewards.to_csv(reward_file)
    # with open(model_path + '/exploration_rate', 'w') as exploration:
    #     exploration.writelines([str(agent.exploration_rate)])
    plt.scatter(range(1, len(rewards) + 1), rewards)
    plt.title('Reward achieved over training episodes')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(model_path + f'reward_plot')
    plt.show()

#
#
# model_path = r"/content/drive/My Drive/AchtungDieKurve/models/colab_tryout_model_0_1_loss"
# training_sessions = 1000
# game = TrainingEnv(['r', 'r'], training_mode=True)
# agent = DQNAgent(game)
# agent.set_model(build_cnn_model(agent.get_state_shape(), agent.get_action_shape()))
# # with open(model_path + '/model_architecture.txt', 'r') as json_file:
# #     config = json.load(json_file)
# #     model = model_from_json(config)
# #     model.load_weights(model_path + f'/session_400_weights')
# #     model.compile(optimizer='Adam', loss = 'mse')
# #     agent.set_model(model)
# with open(model_path + '/model_architecture.txt', 'w') as json_file:
#     config = agent.to_json()
#     json.dump(config, json_file)
# # model.load_weights(model_path + f'/session_{133}_weights')
# # with open(model_path + '/exploration_rate') as exploration:
# #     exp = exploration.readlines()
# #     agent.exploration_rate = float(exp[0])
# # accum_rewards = pd.read_csv(model_path + '/rewards.csv')
# accum_rewards = pd.DataFrame()
# actions_per_game = pd.DataFrame()
# for i in tqdm(range(training_sessions)):
#     if i == 1:
#         with open(model_path + '/model_architecture.txt', 'r') as json_file:
#             config = json.load(json_file)
#             model = model_from_json(config)
#         game.set_player(1, model_path + '/model_architecture.txt',model_path + f'/session_1_weights' )
#         game.players[1]._net = model
#     if i >= 1:
#         game.players[1]._net.load_weights(model_path + f'/session_{i}_weights').expect_partial()
#         with open(model_path + '/exploration_rate') as exploration:
#             exp = exploration.readlines()
#             agent.exploration_rate = float(exp[0])
#     rewards, num_actions = agent.train(2, None, 128)
#     accum_rewards = accum_rewards.append(rewards)
#     actions_per_game = actions_per_game.append(num_actions)
#     agent.save_weights(model_path + f'/session_{i + 1}_weights')
#     with open(model_path + '/rewards.csv','w') as reward_file:
#         accum_rewards.to_csv(reward_file)
#     with open(model_path + '/num_actions.csv', 'w') as action_file:
#         actions_per_game.to_csv(action_file)
#     with open(model_path + '/exploration_rate', 'w') as exploration:
#         exploration.writelines([str(agent.exploration_rate)])
# plt.scatter(range(1, len(accum_rewards) + 1), accum_rewards)
# plt.title('Reward achieved over training episodes')
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.savefig(model_path + f'reward_plot')
# plt.show()


# from double_dqn.agent import gym_DQNAgent
# from tqdm import tqdm
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, ReLU, Dropout, Flatten
# from tensorflow.keras.models import load_model, model_from_json
# from matplotlib import pyplot as plt
# import json
# import pandas as pd
# import numpy as np
# import gym
# import os
#
#
# def build_cnn_model(input_shape, output_shape) -> Model:
#     n_actions = output_shape[0]
#     input_tensor = Input(shape=input_shape)
#     conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
#     max = MaxPool2D(padding='same')(conv)
#     relu = ReLU()(max)
#     for i in range(8):
#         conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(relu)
#         relu = ReLU()(conv)
#         dropout = Dropout(0.3)(relu)
#     flat = Flatten()(dropout)
#     fc = Dense(100, activation='relu')(flat)
#     fc = Dense(100, activation='relu')(fc)
#     final = Dense(n_actions)(fc)
#     model = Model(inputs=input_tensor, outputs=final)
#     model.compile(optimizer='adam', loss='mse')
#     return model
#
#
# def build_fc_model(input_shape, output_shape, num_layers=10) -> Model:
#     n_actions = output_shape.n
#     input_tensor = Input(shape=input_shape)
#     fc = Dense(60, activation='relu')(input_tensor)
#     fc = Dense(40, activation = 'relu')(fc)
#     final = Dense(n_actions)(fc)
#     fc_model = Model(inputs=input_tensor, outputs=final)
#     fc_model.compile(optimizer='adam', loss='mse')
#     return fc_model
#
#
# # model_path = r"C:\Users\danie\Studies\B.Sc\year3\Semester B\67842 - Introduction to Artificial Intelligence\Project\AchtungDeKurve\models\cnn_model"
# this_file_dir = os.path.dirname(__file__)
# project_dir = os.path.join(this_file_dir, os.pardir)
# model_rel_path = "models" + os.path.sep + "fc_model"
# model_path = os.path.join(project_dir, model_rel_path)
#
# if __name__ == "__main__":
#     game = gym.make('MountainCar-v0')
#     training_sessions = 2
#     agent = gym_DQNAgent(game)
#     agent.set_model(build_fc_model(agent.get_state_shape(), agent.get_action_shape()))
#
#     with open(model_path + '/model_architecture.txt', 'w') as json_file:
#         config = agent.to_json()
#         json.dump(config, json_file)
#
#     accum_rewards = pd.DataFrame()
#     actions_per_game = pd.DataFrame()
#     for i in tqdm(range(training_sessions)):
#         rewards, num_actions = agent.train(3000, None, 128)
#         print('average rewards = ', np.average(rewards))
#         accum_rewards = accum_rewards.append(rewards)
#         actions_per_game = actions_per_game.append(num_actions)
#         agent.save_weights(model_path + f'/session_{i + 1}_weigfinal message = hts')
#         with open(model_path + '/rewards.csv', 'w') as reward_file:
#             accum_rewards.to_csv(reward_file)
#         with open(model_path + '/num_actions.csv', 'w') as reward_file:
#             accum_rewards.to_csv(reward_file)
#         with open(model_path + '/exploration_rate', 'w') as exploration:
#             exploration.writelines([str(agent.exploration_rate)])
#     plt.scatter(range(1, len(accum_rewards) + 1), accum_rewards)
#     plt.title('Reward achieved over training episodes')
#     plt.xlabel('episode')
#     plt.ylabel('reward')
#     plt.savefig(model_path + f'reward_plot')
#     plt.show()

#
#
# model_path = r"/content/drive/My Drive/AchtungDieKurve/models/colab_tryout_model_0_1_loss"
# training_sessions = 1000
# game = TrainingEnv(['r', 'r'], training_mode=True)
# agent = DQNAgent(game)
# agent.set_model(build_cnn_model(agent.get_state_shape(), agent.get_action_shape()))
# # with open(model_path + '/model_architecture.txt', 'r') as json_file:
# #     config = json.load(json_file)
# #     model = model_from_json(config)
# #     model.load_weights(model_path + f'/session_400_weights')
# #     model.compile(optimizer='Adam', loss = 'mse')
# #     agent.set_model(model)
# with open(model_path + '/model_architecture.txt', 'w') as json_file:
#     config = agent.to_json()
#     json.dump(config, json_file)
# # model.load_weights(model_path + f'/session_{133}_weights')
# # with open(model_path + '/exploration_rate') as exploration:
# #     exp = exploration.readlines()
# #     agent.exploration_rate = float(exp[0])
# # accum_rewards = pd.read_csv(model_path + '/rewards.csv')
# accum_rewards = pd.DataFrame()
# actions_per_game = pd.DataFrame()
# for i in tqdm(range(training_sessions)):
#     if i == 1:
#         with open(model_path + '/model_architecture.txt', 'r') as json_file:
#             config = json.load(json_file)
#             model = model_from_json(config)
#         game.set_player(1, model_path + '/model_architecture.txt',model_path + f'/session_1_weights' )
#         game.players[1]._net = model
#     if i >= 1:
#         game.players[1]._net.load_weights(model_path + f'/session_{i}_weights').expect_partial()
#         with open(model_path + '/exploration_rate') as exploration:
#             exp = exploration.readlines()
#             agent.exploration_rate = float(exp[0])
#     rewards, num_actions = agent.train(2, None, 128)
#     accum_rewards = accum_rewards.append(rewards)
#     actions_per_game = actions_per_game.append(num_actions)
#     agent.save_weights(model_path + f'/session_{i + 1}_weights')
#     with open(model_path + '/rewards.csv','w') as reward_file:
#         accum_rewards.to_csv(reward_file)
#     with open(model_path + '/num_actions.csv', 'w') as action_file:
#         actions_per_game.to_csv(action_file)
#     with open(model_path + '/exploration_rate', 'w') as exploration:
#         exploration.writelines([str(agent.exploration_rate)])
# plt.scatter(range(1, len(accum_rewards) + 1), accum_rewards)
# plt.title('Reward achieved over training episodes')
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.savefig(model_path + f'reward_plot')
# plt.show()
