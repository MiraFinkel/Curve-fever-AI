from game.training_environment import TrainingEnv
from game.settings import *
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from double_dqn.agent import DQNAgent
from tqdm import tqdm
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, ReLU, Dropout, Flatten, Concatenate, Multiply, Add, BatchNormalization
from tensorflow.keras.models import load_model, model_from_json
from matplotlib import pyplot as plt
import json
import pandas as pd


def build_cnn_model(input_shape, output_shape) -> Model:
    n_actions = output_shape[0]
    input_tensor = Input(shape=input_shape)
    angle = Input(shape=(14,))
    conv = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(input_tensor)
    max = MaxPool2D(padding='same')(conv)
    relu = ReLU()(max)
    normalized = BatchNormalization()(relu)

    conv = Conv2D(filters=8, kernel_size=(3, 3), padding='same')(normalized)
    relu = ReLU()(conv)
    normalized = BatchNormalization()(relu)

    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(normalized)
    relu = ReLU()(conv)
    normalized = BatchNormalization()(relu)

    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(normalized)
    relu = ReLU()(conv)
    normalized = BatchNormalization()(relu)

    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(normalized)
    relu = ReLU()(conv)
    normalized = BatchNormalization()(relu)

    flat = Flatten()(normalized)
    # concat = Concatenate()([flat, angle])
    fc = Dense(100, activation='relu')(flat)
    angle_factors = Dense(100, use_bias=False)(angle)
    fc_angled = Add()([fc, angle_factors])
    fc_angled = Dense(100, activation='relu')(fc_angled)
    dropout = Dropout(0.5)(fc_angled)
    final = Dense(n_actions)(dropout)
    model = Model(inputs=[input_tensor, angle], outputs=final)
    model.compile(optimizer='adam', loss='mse')
    return model


# model_path = r"C:\Users\danie\Studies\B.Sc\year3\Semester B\67842 - Introduction to Artificial Intelligence\Project\AchtungDeKurve\models\cnn_model"
this_file_dir = os.path.dirname(__file__)
project_dir = os.path.join(this_file_dir, os.pardir)
model_rel_path = "models" + os.path.sep + "cnn_model2"
model_path = os.path.join(project_dir, model_rel_path)

start_session = 0
weight_file = 'session_' + str(start_session) + '_weights'

if __name__ == "__main__":
    # Initiate data structures
    training_sessions = 2
    batch_size = 32
    game = TrainingEnv(['r', 'r'], [(), ()], training_mode=True)
    agent = DQNAgent(game)
    agent.set_model(build_cnn_model(agent.get_state_shape(), agent.get_action_shape()))
    if start_session >= 1:
        agent.net.q_net.load_weights(model_path + f'/session_{start_session}_weights')
        agent.net.target_net.load_weights(model_path + f'/session_{start_session}_weights')
    config = agent.to_json()
    accum_rewards = pd.DataFrame()

    # save model architecture
    with open(os.path.join(model_path, 'model_architecture'), 'w') as json_file:
        json.dump(config, json_file)

    for i in tqdm(range(start_session, training_sessions + start_session)):
        Arena.ARENA_WIDTH = 100
        Arena.ARENA_HEIGHT = 100
        if i > 0 and i % 200 == 0:
            Arena.ARENA_WIDTH += 50
            Arena.ARENA_HEIGHT += 50
        if i == start_session + 1:
            # Set player to be reinforcement player
            with open(os.path.join(model_path, 'model_architecture')) as json_file:
                model = model_from_json(json.load(json_file))
                game.set_player(1, 'd', model)
        if i >= start_session + 1:
            # Set players weights to be the trained weights from last session
            game.players[1]._net.load_weights(model_path + f'/session_{i}_weights')

        rewards, num_actions = agent.train(1, None, batch_size)

        # Save all values that need to be saved
        accum_rewards = accum_rewards.append(rewards)
        agent.save_weights(os.path.join(model_path, f'session_{i + 1}_weights'))
        with open(os.path.join(model_path, 'rewards.csv'), 'w') as reward_file:
            accum_rewards.to_csv(reward_file)

    # If we finished all training sessions, we plot the rewards for each episode played
    plt.scatter(range(1, len(accum_rewards) + 1), accum_rewards)
    plt.title('Reward achieved over training episodes')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(os.path.join(model_path, f'reward_plot'))
    plt.show()
