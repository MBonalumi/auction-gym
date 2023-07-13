import numpy as np
from Bidder import Bidder
from BidderBandits import BaseBandit
from ModelsMine import SAC as SAC

################################
######        SAC         ######
################################
class SACBidder(BaseBandit):
    def __init__(self, rng):
        super(SACBidder, self).__init__(rng)
        # self.expected_utilities = np.zeros(5)
        
        self.sac = SAC(state_dim=2, action_dim=1, max_action=3.0, gamma=0.0, rng=rng)
        pass

    def bid(self, value, context, estimated_CTR):
        #   TODO
        #   call action
        state = np.array((value, estimated_CTR))
        chosen_bid = self.sac.select_action(state)
        chosen_bid = chosen_bid[0]
        pass
        return chosen_bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)   #only used for regrets!
        surpluses[won_mask] = np.array((values[won_mask] * outcomes[won_mask]) - prices[won_mask])
        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!
        
        
        # what i do here:
        #   i need the S,A,R,S' tuples
        #   i have S,A
        #   so i use S = S[:-1]   and   S' = S[1:]
        #   I AM WASTING A SAMPLE OUT OF 10 EVERY TIME THOUGH!!!
        #   TODO: i could make up the last s' e.g. =0 since i put gamma=0
        
        # def add(self, state, action, reward, next_state, done):
        # !!! next_states = ???

        states = [(v,ctr) for (v,ctr) in zip(values[:-1], estimated_CTRs[:-1])]
        next_states = [(v,ctr) for (v,ctr) in zip(values[1:], estimated_CTRs[1:])]
        
        actions = np.array(bids[:-1])
        done = False
        values = values[:-1]
        prices = prices[:-1]
        outcomes = outcomes[:-1]
        won_mask = won_mask[:-1]
        rewards = np.zeros_like(values)     # same as surpluses, but translated for learning
        rewards[won_mask] = np.array((values[won_mask] * outcomes[won_mask]) - prices[won_mask])
        
        
        pass
        #   add to sample all stuff
        for i in range(values.size):
            pass
            self.sac.replay_buffer.add(state=states[i],
                                       action=actions[i],
                                       reward=rewards[i],
                                       next_state=next_states[i],
                                       done=done)
        
        #   train
        self.sac.replay_buffer
        self.sac.train(100)

    def clear_logs(self, memory):
        #   TODO: boh!
        pass


#################################
#####   Stable Baselines   ######
#################################

from stable_baselines3 import SAC as sb3SAC, PPO as sb3PPO
# import gymnasium as gym
from ModelsMine import BidEnv

# Using PPO since SAC doesn't support Discrete ActionSpace but only Box (???) 

class SB3_Bidder_discrete(BaseBandit):      # Stable Baselines 3 version of the bidder
    def __init__(self, rng):
        super(SB3_Bidder_discrete, self).__init__(rng)
        self.env = BidEnv(rng=rng, num_bids=self.NUM_BIDS)
        self.model = sb3PPO(policy='MlpPolicy', env=self.env, gamma=0.0, device='cpu')  # cpu should be faster

    def bid(self, value, context, estimated_CTR):
        bid_index = self.model.predict(observation = context) [ 0 ]
        chosen_bid = self.BIDS[bid_index]
        return chosen_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        rewards = np.zeros_like(values)
        rewards[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, rewards)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!

        # train
        self.env.actions_rewards.extend( np.stack((bids, rewards), axis=1) )
        self.env.observations.extend(contexts)

        # steps = 1000 or 3*samples if less than 1000
        timesteps = min(100, actions_rewards.shape[1] * 3)
        '''
        100 steps is TOO SMALL!
        '''
        self.model.learn(total_timesteps=timesteps)

from gymnasium import spaces
class SB3_Bidder_continuous(SB3_Bidder_discrete):      # Stable Baselines 3 version of the bidder
    def __init__(self, rng):
        super(SB3_Bidder_continuous, self).__init__(rng)
        self.env.action_space = spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32) # generate bid in [0,3]
        self.model = sb3SAC(policy='MlpPolicy', env=self.env, gamma=0.0, device='cpu')  # cpu should be faster

    def bid(self, value, context, estimated_CTR):
        return self.model.predict(observation = context) [ 0 ][ 0 ]
    
'''
    ### why to use device='cpu' instead of device='cuda' ###
    ( https://github.com/DLR-RM/stable-baselines3/issues/350 )

I assume your custom environment has small observations/actions
(1D vectors of few hundred items).
In this case running code on pure CPU (device=cpu) can be faster
as calling CUDA operations becomes a bottleneck at small scales.

Note: seems to be faster! sb3_discr-vs-cont ~20min/run instead of ~45min/run 
'''