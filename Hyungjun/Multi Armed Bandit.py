#!/usr/bin/env python
# coding: utf-8

# # Bandit
# for first step of reinforcement learning, implement multi-armed bandit (MAB) problems

# In[1]:


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# # 멀티암드밴딧 (Multi Armed Bandit)
# 하나의 밴딧 (예를 들면 슬롯 머신) 에서, 여러 손잡이가 존재 (Multi-Armed)
# 여러 손잡이들은 각각 다른 비율로 양(+1) 또는 음(-1)의 보상을 부여
# 목적은 에이전트가 가장 보상이 높을 Arm을 자주 선택하게 되는 것
# 모든 의도와 목적에 대해 불변하는 오직 한 개의 상태만이 존재, 해당 상태에서의 행동도 고정
# 즉, 환경의 상태를 완전히 무시하는 에이전트 설계
# 

# In[7]:


# 밴딧의 손잡이 목록을 작성
# 현재 손잡이들 중 4번째(인덱스는 3)손잡이가 가장 자주 양의 보상을 제공토록 설정
bandit_arms = [0.2,0,-0.2,-2] # 음의 값인 이유는 랜덤하게 생성된 값이 이 값보다 크게 나올 확률이 음일 경우 높아지므로

num_arms = len(bandit_arms)
def pullBandit(bandit):
    # 랜덤한 값을 구함````
    result = np.random.randn(1)
    if result > bandit:
        # 양의 보상을 반환
        return 1
    else:
        # 음의 보상을 반환
        return -1


# In[8]:


tf.reset_default_graph()

# 네트워크의 피드포워드 부분을 구현한다.
weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

# 학습 과정을 구현
# 보상과 선택된 액션을 네트워크에 피드해줌으로써 비용을 계산하고
# 비용을 이용해 네트워크를 업데이트한다.
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_output = tf.slice(output,action_holder,[1])
loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)


# In[9]:


# 에이전트를 학습시킬 총 에피소드의 수를 설정한다.
total_episodes = 1000

# 밴딧 손잡이에 대한 점수판을 0으로 설정
total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer()

# 텐서플로 그래프를 론칭한다.
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        # 볼츠만 분포에 따라 액션 선택
        actions = sess.run(output)
        a = np.random.choice(actions, p=actions)
        action = np.argmax(actions == a)
        
        # 밴딧 손잡이 중 하나를 선택함으로써 보상을 받는다.
        reward = pullBandit(bandit_arms[action])
        
        # 네트워크를 업데이트한다
        _,resp,ww = sess.run([update,responsible_output,weights],                            feed_dict={reward_holder:[reward],action_holder:[action]})
        
        # 보상의 층계 업데이트
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
        
        i+= 1
        
    print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising...")
    if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
        print("...and it was right")
    else:
        print("...and it was wrong")


# In[ ]:




