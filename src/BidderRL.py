import numpy as np
from Bidder import Bidder
from ModelsMine import SAC

################################
######        SAC         ######
################################
class SACBidder(Bidder):
    def __init__(self, rng):
        super(SACBidder, self).__init__(rng)
        self.expected_utilities = np.zeros(5)
        
        self.sac = SAC(state_dim=2, action_dim=1, max_action=3.0)
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
        rewards = np.zeros_like(values)
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
