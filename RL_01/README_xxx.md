# 01 Reinforcement Learning 

## Deep Q-Learning


## Deep Deterministic Policy Gradient
DDPG implement DPG algorithm with actor-critic parametrized with neural networks.

### Actor ###
Actor estimate policy $\pi_\theta(s)$ which maps from state to action. 

### Critic ###
Critic estimate action value function under policy $\pi_\theta(s)$ $Q_\pi_\theta(s,a)$.


### My results
<p align="center">![DDPG Performance on swing up pendulum (Pendulum-v0)](./02_DDPG/Trained_Models/DDPG_Pendulum-v0/reward_vs_episode.png)</p>
