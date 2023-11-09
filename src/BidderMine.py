'''
contains the bidder classes for the thesis experiments
'''

import numpy as np
import pandas as pd
from BidderBandits import BaseBidder
from BidderNovelty import NoveltyBidder
from utils import get_project_root, parse_kwargs

ROOT_DIR = get_project_root()

### MY PROPOSED ALG ###
class ProposedAlg(BaseBidder):
    def __init__(self, rng, value_obj, arms, n_context, gamma=1.0):
        super(ProposedAlg, self).__init__(rng)
        self.n_context = n_context
        self.t = 1
        self.value_obj = value_obj
        self.arms = arms
        self.n_actions = len(arms)
        self.gamma = gamma

        #ctr
        self.N_buy = np.zeros(self.n_context, dtype=int)
        self.N_win = np.zeros(self.n_context, dtype=int)

        #arms
        self.N_win_a = np.ones((self.n_context, self.n_actions), dtype=int)
        self.N_play_a = np.ones((self.n_context, self.n_actions), dtype=int)
        
        self.obj_fun = lambda value, ctr, bid, win_prob:    (value * ctr - bid) * win_prob
        
        self.last_action = None
        self.last_context = None

    def alg_bid(self, value, context_i):
        ucb_ctr = self.N_buy[context_i] / self.N_win[context_i] + self.gamma * np.sqrt( np.log(self.t) / self.N_win[context_i] )
        ucbs_win_prob = self.N_win_a[context_i, :] / self.N_play_a[context_i, :] + self.gamma * np.sqrt( np.log(self.t) / self.N_play_a[context_i, :] )

        if np.isnan(ucb_ctr): ucb_ctr = 1.
        ucbs_win_prob[np.isnan(ucbs_win_prob)] = np.inf

        ucbs = self.obj_fun(value, ucb_ctr, self.arms, ucbs_win_prob)

        self.last_action = np.argmax(ucbs)
        self.last_context = context_i
        return self.arms[self.last_action]

    def bid(self, value, context, estimated_CTR):
        contexts_set = np.array([-1.09, 0.0, 1.09], dtype=np.float32)[:self.n_context]
        context_i = np.abs(contexts_set - context[0]).argmin()
        # context_i = np.array(np.where(contexts_set == context[0]), dtype=int).squeeze()
        return np.float32( self.alg_bid(value, context_i) )

    def alg_update(self, has_won, has_buy):
        self.t += 1
        
        self.N_buy[self.last_context] = self.N_buy[self.last_context] + has_buy
        self.N_win[self.last_context] = self.N_win[self.last_context] + has_won
        self.N_win_a[self.last_context, self.last_action] = self.N_win_a[self.last_context, self.last_action] + has_won
        self.N_play_a[self.last_context, self.last_action] = self.N_play_a[self.last_context, self.last_action] + 1

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        has_win = won_mask[0].astype(int)
        has_buy = outcomes[0].astype(int) * has_win
        self.alg_update(has_win, has_buy)

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)


### UCB1 ###
class UCB1_new(BaseBidder):
    def __init__(self, rng, C=2**0.5):
        super(UCB1_new, self).__init__(rng)
        self.t = 1

        self.C = C

        self.counters = np.zeros(self.NUM_BIDS)
        self.exp_utility = np.zeros(self.NUM_BIDS)
        self.ucbs = np.ones(self.NUM_BIDS) * np.inf

    def bid(self, value, context, estimated_CTR):
        max_ucb_bids = self.BIDS[self.ucbs == self.ucbs.max()]
        chosen_bid = self.rng.choice(max_ucb_bids)
        return chosen_bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        self.t += values.size
        
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        played_bids = np.unique(bids)
        for bid in played_bids:
            mask = bids == bid
            bid_surpluses = surpluses[mask]
            n_plays = bid_surpluses.size
            i = np.where(self.BIDS == bid)[0][0]
            self.expected_utilities[i] = (self.expected_utilities[i] * self.counters[i] + bid_surpluses.sum()) / (self.counters[i] + n_plays)
            self.counters[i] += n_plays
            self.ucbs[i] = self.expected_utilities[i] + self.C * np.sqrt(np.log(self.t) / self.counters[i])
            
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)


### EXP3 ###
class Exp3_new(BaseBidder):
    def __init__(self, rng, gamma=0.05, step=0.1):
        super(Exp3_new, self).__init__(rng)
        self.t = 1
        self.gamma = gamma   # gamma = cubic_root( (5 * ln5)/(2 * 118'000) ) = 0.0324
        self.step = step    # could be sqrt(gamma) or 2*gamma too

        self.exp_utility = np.zeros(self.NUM_BIDS)
        self.w = np.ones(self.NUM_BIDS)
        self.p = np.ones(self.NUM_BIDS, dtype=np.float64) / self.NUM_BIDS
        self.p[0] = 1 - self.p[1:].sum()    # make sure that sum(p)=1 

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        self.t += len(values)
        
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        rewards = surpluses / values       # rewards are normalized, in [0,1]

        for i, bid in enumerate(bids):
            arm_id = np.where(self.BIDS == bid)[0][0]
            self.exp_utility[arm_id] += rewards[i] / np.sqrt(self.p[arm_id])
            self.w[arm_id] = np.exp(self.exp_utility[arm_id] / self.NUM_BIDS * self.step)
            self.w[~np.isfinite(self.w)] = 0    # deactivate arms with infinite weight
            # self.p = (1 - self.gamma/10) * self.w / self.w.sum()  +  self.gamma/10 / self.NUM_BIDS
            self.p = self.w / self.w.sum()
        
        self.p = self.p / self.p.sum()
        self.p[np.argmax(self.p)] = 1 - (np.sum(self.p) - np.max(self.p))

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

    def bid(self, value, context, estimated_CTR):
        pulled_arm = self.rng.choice(self.NUM_BIDS, p=self.p)
        return self.BIDS[pulled_arm]


### EXP3 implementazione Marco base ###
class Exp3_marcobase(BaseBidder):
    def __init__(self, rng, gamma=0.1, obj_value=1, add_factor=0, random_state=1):
        super(Exp3_marcobase, self).__init__(rng)
        self.gamma = gamma
        self.obj_value = obj_value
        self.add_factor = add_factor
        self.random_state = random_state
        self.n_arms = len(self.BIDS)

        self.w = np.ones(self.n_arms)
        self.est_rewards = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])

    def bid(self, value, context, estimated_CTR):
        self.last_pull = np.random.choice(  np.arange(self.n_arms),
                                            p=self.probabilities,
                                            size=None)
        return self.BIDS[self.last_pull]
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # surpluses = np.zeros_like(values)
        # surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        surplus = values[-1] * outcomes[-1] - prices[-1] if won_mask[-1] else 0
        reward = (surplus + self.add_factor) / self.obj_value

        self.est_rewards[self.last_pull] = reward / self.probabilities[self.last_pull]
        self.w[self.last_pull] *= np.exp(self.gamma * self.est_rewards[self.last_pull] / self.n_arms)
        self.w[~np.isfinite(self.w)] = 0
        # self.probabilities = (1 - self.gamma) * self.w / sum(self.w) + self.gamma / self.n_arms
        self.probabilities = self.w / sum(self.w)
        # self.probabilities[0] = 1 - sum(self.probabilities[1:])
        #   instead of putting the remainder to first arm, put it on the highest prob arm, 
        #   so to avoid negative prob if p[0] is very low
        self.probabilities[np.argmax(self.probabilities)] = 1 - (np.sum(self.probabilities) - np.max(self.probabilities))

        
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
    

### EXP3 implementazione Marco ###
class Exp3_marco(BaseBidder):
    def __init__(self, rng, gamma=0.1, eta=1, obj_value=1, add_factor=0, random_state=1):
        super(Exp3_marco, self).__init__(rng)
        self.gamma = gamma
        self.eta = eta
        self.obj_value = obj_value
        self.add_factor = add_factor
        self.random_state = random_state
        self.n_arms = len(self.BIDS)

        self.G = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])

    def bid(self, value, context, estimated_CTR):
        self.probabilities /= self.probabilities.sum()
        self.last_pull = np.random.choice(  np.arange(self.n_arms),
                                            p=self.probabilities,
                                            size=None)
        return self.BIDS[self.last_pull]
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surplus = values[-1] * outcomes[-1] - prices[-1] if won_mask[-1] else 0
        reward = (surplus + self.add_factor) / self.obj_value

        reward = (reward + self.add_factor) / self.obj_value
        reward_vect = np.zeros(self.n_arms)
        reward_vect[self.last_pull] = reward / self.probabilities[self.last_pull]
        self.G = self.G + reward_vect
        # div = np.sum(np.array([np.exp(self.eta * self.G[i])   for i in range(self.n_arms)]))
        # for i in range(self.n_arms):
        #     self.probabilities[i] = np.exp(self.eta * self.G[i]) / div
        self.probabilities = np.exp(self.eta * self.G) / np.exp(self.eta * self.G).sum()
        
        self.probabilities[np.argmax(self.probabilities)] = 1 - np.sum(self.probabilities[np.delete(np.arange(self.n_arms), np.argmax(self.probabilities))])
        #   sometimes best action has inf prob, and p/p.sum() makes it nan, 
        #   this solves
        
        # self.probabilities = (1 - self.gamma) * self.probabilities + self.gamma / self.n_arms
        
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)



### PseudoExpert w UCB1 or Exp3 ###
class PseudoExpert_new(BaseBidder):
    def __init__(self, rng, sub_bidder=UCB1_new, sub_bidder_kwargs={}):
        super(PseudoExpert_new, self).__init__(rng, )
        self.rng = rng
        self.sub_bidder_type = sub_bidder
        self.sub_bidder_kwargs = sub_bidder_kwargs
            # for exp3, lr must be multiplied for cbrt(3),
            #   because the number of iterations for each subbidder is 1/3 of the total
            #   so 0.0467 instead of 0.0324
        self.sub_bidders = []
        self.counters = []
        self.contexts_set = []

    def bid(self, value, context, estimated_CTR):
        c = context[0]
        if c not in self.contexts_set:
            self.contexts_set.append(c)
            new_sub_bidder = eval(f"self.sub_bidder_type(rng=self.rng {parse_kwargs(self.sub_bidder_kwargs)})")
            new_sub_bidder.total_num_auctions = self.total_num_auctions
            new_sub_bidder.num_iterations = self.num_iterations
            new_sub_bidder.agent_id = len(self.contexts_set) - 1
            self.sub_bidders.append(new_sub_bidder)
            self.counters.append(0)
        
        i_context = self.contexts_set.index(c)
        self.counters[i_context] += 1
        return self.sub_bidders[i_context].bid(value, context, estimated_CTR)
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        
        # when one auction per iteration
        context = contexts[0]
        i_ctxt = np.where(self.contexts_set == context[0])[0][0]
        self.sub_bidders[i_ctxt].winning_bids = self.winning_bids
        self.sub_bidders[i_ctxt].second_winning_bids = self.second_winning_bids
        self.sub_bidders[i_ctxt].update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        
        # for i_ctxt, ctxt in enumerate(self.contexts_set):
        #     ctxt_mask = np.array((contexts[:,0] == ctxt)).squeeze()
        #     if ctxt_mask.sum() == 0: continue
        #     self.sub_bidders[i_ctxt].winning_bids = self.winning_bids[ctxt_mask]
        #     self.sub_bidders[i_ctxt].second_winning_bids = self.second_winning_bids[ctxt_mask]
        #     self.sub_bidders[i_ctxt].update(  contexts[ctxt_mask], values[ctxt_mask], bids[ctxt_mask], prices[ctxt_mask], outcomes[ctxt_mask],
        #                                         estimated_CTRs[ctxt_mask], won_mask[ctxt_mask], iteration, plot, figsize, fontsize, name)
            
        if iteration == self.num_iterations - 1:
            print(self.sub_bidder_type)
            print(self.counters)
            for i_ctxt, ctxt in enumerate(self.contexts_set):
                ctxt_bids = np.array(self.sub_bidders[i_ctxt].bids)
                print(f"{i_ctxt}. value:{ctxt} shape:{ctxt_bids.shape}")
                ctxt_bids_df = pd.DataFrame(ctxt_bids, columns=["bid"])
                print(ctxt_bids_df.value_counts(), '\n\n', end="")
