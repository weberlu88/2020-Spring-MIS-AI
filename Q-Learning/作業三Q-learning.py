

"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
"""

#-------import------------
import numpy as np
import pandas as pd
import time

#-------global variable---------
N_STATES = 6   #有多少種states
ACTIONS = ['left', 'right']     #可以做的動作
EPSILON = 0.9   # epsilon greedy
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

#--------建立Q table----------
def build_q_table(n_states, actions): 
    table = pd.DataFrame(np.zeros((n_states, len(actions))),columns=actions,)   
    return table

#--------choose action的功能-----------
def choose_action(state, q_table): 
    state_actions = q_table.iloc[state, :] #取state這一行的對應資料
    #act non-greedy or state-action have no value
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  
        action_name = np.random.choice(ACTIONS) 
    else:   # act greedy
        action_name = state_actions.idxmax()
    return action_name

#--------建立環境對我們行為的feedback---------
def get_env_feedback(S, A): 
    if A == 'right':    # move right
        if S == N_STATES - 2:   #寶藏前一個位置
            S_ = 'terminal'  
            R = 1 #找到才給reward
        else: 
            S_ = S + 1 
            R = 0 
    else:   # move left
        R = 0
        if S == 0: 
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

#-----------更新環境--------------
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment 
    if S == 'terminal': 
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter) #回應
        print('\r{}'.format(interaction), end='') 
        time.sleep(2)                             
        print('\r                                ', end='') #清空
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list) 
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

#----------建立reinforcement learning-----------
def rl():
    q_table = build_q_table(N_STATES, ACTIONS) #建立 Q table
    for episode in range(MAX_EPISODES): #從第一個回合玩到最後一個回合
        step_counter = 0
        S = 0 #初始情況，探索者放到左邊
        is_terminated = False 
        update_env(S, episode, step_counter) #更新環境
        while not is_terminated: #回合沒有結束

            A = choose_action(S, q_table) 
            S_, R = get_env_feedback(S, A)  
            q_predict = q_table.loc[S, A] #估計值
            if S_ != 'terminal': #回合還沒結束
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #真實值 
            else:
                q_target = R    
                is_terminated = True    # 結束這一回合
                
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

