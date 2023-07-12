import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Concatenate, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.special import softmax

class ActorCriticDense:
    """
        action_dim, df_activity, lr_actor=0.001, lr_critic=0.001, gamma=0.99, 
        actor_filepath="", critic_filepath="", 
        geological_position=[45, -90], mbti_type="ISTJ", age=50
    """
    def __init__(self, action_dim, df_activity, user_item, ffm_data=5*[0.5], lr_actor=0.0001, lr_critic=0.0001, gamma=0.99, 
                 actor_filepath="", critic_filepath="", 
                 geological_position=[45, -90], age=50):
        
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.df_activity = df_activity
        self.gamma = gamma
        self.sbert_size = len(df_activity["Embedding"][0])
        self.geological_position = [(geological_position[0]+90)/180, (geological_position[0]+180)/360]
        self.ffm_data = ffm_data
        self.age = age/122 #minmax scale to oldest ever age
        self.state_dim = len(df_activity["Embedding"][0]) + len(self.ffm_data) + len(geological_position) + 1
        self.action_dim = action_dim
        self.prev_action = -10
        self.user_item = user_item

        if actor_filepath:
            self.actor = self.load_actor(actor_filepath)
        else:    
            self.actor = self.build_actor()
        if critic_filepath:
            self.critic = self.load_critic(critic_filepath)
        else:    
            self.critic = self.build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr_critic)

    def build_actor(self):
        
        user_item = Input(shape=(self.action_dim))
        lstm_layer1 = Dense(64, activation='relu')(user_item)
        lstm_layer2= Dense(48, activation='relu')(lstm_layer1)
#
        state_input = Input(shape=(self.state_dim))
        
        merged = Concatenate(axis=1)([state_input, lstm_layer2])
        norm = BatchNormalization()(merged)
        dense3 = Dense(self.action_dim*2, activation='relu')(norm)
        output_layer = Dense(self.action_dim, activation='softmax')(dense3)
        model = Model(inputs=[state_input, user_item], outputs=[output_layer])

        return model

    def build_critic(self):
        user_item = Input(shape=(self.action_dim))
        dense1 = Dense(64, activation='relu')(user_item)
        dense2 = Dense(48, activation='relu')(dense1)

        state_input = Input(shape=(self.state_dim))
        
        merged = Concatenate(axis=1)([state_input, dense2])
        norm = BatchNormalization()(merged)
        dense3 = Dense(self.action_dim*2, activation='relu')(norm)
        output_layer = Dense(1, activation='linear')(dense3)
        model = Model(inputs=[state_input, user_item], outputs=output_layer)
        
        return model
    
    def save_critic(self, filepath="critic_model.h5"):
        self.critic.save(filepath)
    
    def load_critic(self, filepath="critic_model.h5"):
        return tf.keras.models.load_model(filepath)
    
    def save_actor(self, filepath="actor_model.h5"):
        self.actor.save(filepath)
    
    def load_actor(self, filepath="actor_model.h5"):
        return tf.keras.models.load_model(filepath)

    def get_actions(self, state_idx, batch_size, max_prob=None):
        actions = []
        states = [state_idx]
        self.prev_action = state_idx
        state = self.state_to_embedding(state_idx)
        for i in range(batch_size):
            action_prob = self.actor.predict([state, self.user_item], verbose=0)
            if isinstance(max_prob, list):
                max_prob.append(np.amax(action_prob))
            action = np.random.choice(self.action_dim, p=action_prob[0])
            if i < batch_size - 1:
                states.append(action)
            actions.append(action)
            state = self.state_to_embedding(action)
            
        return states, actions
    
    def state_to_embedding(self, state_idx):
        state = np.zeros((1, self.state_dim))
        state[0, :self.sbert_size] = np.asarray(MinMaxScaler(self.df_activity["Embedding"][state_idx]).feature_range)
        state[0, self.sbert_size:self.sbert_size+5] = self.ffm_data
        state[0, -3:-1] = self.geological_position
        state[0, -1] = self.age
        return state
    


    def update_user_item(self, action, reward):
        self.user_item[0][action] = reward


    def train(self, state_lst, action_lst, reward_lst):
        state_lst = np.array(state_lst)
        action_lst = np.array(action_lst)
        reward_lst = np.array(reward_lst)

        for i in range(len(state_lst)):
            state = np.zeros(( 1, self.state_dim))
            state[0, :] = self.state_to_embedding(state_lst[i])
            next_state = np.zeros((1, self.state_dim))
            next_state[0, :] = self.state_to_embedding(action_lst[i])
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                action_probs= self.actor([state, self.user_item])

                values = tf.squeeze(self.critic([state, self.user_item]))
                self.update_user_item(action_lst[i], reward_lst[i])
                next_state_values = tf.squeeze(self.critic([next_state, self.user_item]))

                # Compute advantages
                # returns = np.zeros_like(reward_lst)
                advantage = np.zeros_like(reward_lst)
                # G = 0
                # for t in reversed(range(len(reward_lst))):
                #     G = self.gamma * G + reward_lst[t]
                #     returns[t] = G
                #     td_error = G - values[t]
                #     advantage[t] = td_error
                advantage = reward_lst + self.gamma*next_state_values - values
                # Normalize advantages

                # Compute actor loss
                actions_onehot = tf.one_hot(action_lst, self.action_dim, dtype=tf.float32)
                log_probs = tf.math.log(tf.reduce_sum(action_probs * actions_onehot, axis=1))
                actor_loss = -tf.reduce_sum(log_probs * advantage)

                # Compute critic loss
                critic_loss = tf.reduce_mean(tf.square(advantage))

            # Update actor and critic
            actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return
