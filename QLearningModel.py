from tensorflow import stack
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, GRU
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.special import softmax

class DeepQLearningAgent:
    def __init__(self, action_size, df_activity, user_item, ffm_data=5*[0.5], buffer_size=20, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 geological_position=[45, -90], age=50):
        self.df_activity = df_activity
        self.sbert_size = len(df_activity["Embedding"][0])
        self.action_dim = action_size
        self.gamma = gamma  # discount rate
        self.geological_position = [(geological_position[0]+90)/180, (geological_position[0]+180)/360]
        self.ffm_data = ffm_data
        self.age = age/122 #minmax scale to oldest ever age
        self.state_dim = len(df_activity["Embedding"][0]) + len(self.ffm_data) + len(geological_position) + 1
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.user_item = user_item
        self.user_item_buffer = np.repeat(self.user_item, buffer_size, axis=0)
        self.memory = []
        self.model = self._build_model()

    def _build_model(self):
        user_item = Input(batch_input_shape=(1, None, self.action_dim))
        gru_layer1 = GRU(64, activation='relu', stateful=False, return_sequences=True)(user_item)
        gru_layer3= GRU(12, activation='tanh', stateful=False, return_sequences=False)(gru_layer1)

        state_input = Input(batch_input_shape=(1, self.state_dim))
        
        merged = Concatenate(axis=1)([state_input, gru_layer3])
        dense1 = Dense(int(self.state_dim/2), activation='relu')(merged)
        dense2 = Dense(int(self.state_dim/4), activation='relu')(dense1)
        norm = BatchNormalization()(dense2)
        dense3 = Dense(self.action_dim*2, activation='relu')(norm)
        output_layer = Dense(self.action_dim, activation='softmax')(dense3)
        model = Model(inputs=[state_input, user_item], outputs=[output_layer])
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def state_to_embedding(self, state_idx):
        state = np.zeros((1, self.state_dim))
        state[0, :self.sbert_size] = np.asarray(MinMaxScaler(self.df_activity["Embedding"][state_idx]).feature_range)
        state[0, self.sbert_size:self.sbert_size+5] = self.ffm_data
        state[0, -3:-1] = self.geological_position
        state[0, -1] = self.age
        return state
    
    def update_user_item_buffer(self, action, reward):
        self.user_item[0][action] = reward
        self.user_item_buffer = np.roll(self.user_item_buffer,axis=0, shift=-1)
        self.user_item_buffer[-1,:] = self.user_item
    
    def remember(self, state_idx, action_idx, reward):
        self.memory.append((state_idx, action_idx, reward))

    def act(self, state_idx):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = self.state_to_embedding(state_idx)
        act_values = self.model.predict([state, np.expand_dims(self.user_item_buffer, axis=0)])
        return np.argmax(act_values[0])

    def train(self):
        for state_idx, action_idx, reward in self.memory:
            next_state = self.state_to_embedding(action_idx)
            state = self.state_to_embedding(state_idx)
            tmp_user_item_buffer = self.user_item_buffer
            self.update_user_item_buffer(action_idx, reward)
            target = reward + self.gamma * np.amax(self.model.predict([next_state, np.expand_dims(self.user_item_buffer, axis=0)])[0])
            target_f = self.model.predict([state, np.expand_dims(self.user_item_buffer, axis=0)])
            target_f[0][action_idx] = target
            self.model.fit([state, np.expand_dims(tmp_user_item_buffer, axis=0)], target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
