import numpy as np
from Bidder import Bidder
import matplotlib.pyplot as plt


################################
######     Base Bandit    ######
################################
class BaseBandit(Bidder):
    def __init__(self, rng, auction_type='SecondPrice'):
        super(BaseBandit, self).__init__(rng)
        self.auction_type = auction_type

        # Actions -> discrete, finite, fixed
        self.BIDS = np.array([0.005, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.9, 2.4])
        self.NUM_BIDS = self.BIDS.size
        self.counters = np.zeros_like(self.BIDS)

        # Reward -> avg payoff history
        self.expected_utilities = np.zeros_like(self.BIDS)

        self.regret = []
        self.actions_rewards = []       # not used for now
        self.total_reward = 0
        self.total_regret = 0

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        pass

    def bid(self, value, context, estimated_CTR):
        return 0.123456789      # recognizable value
    
    def clear_logs(self, memory):
        pass

    def calculate_regret_in_hindsight(self, arms, values, prices, surpluses):
        '''
        function that calculates
        rewards and regrets in hindsight
        for a batch of auctions
        '''

        rewards_hindsight = np.zeros((values.size, 2))      # tuples (arm, reward)
        
        for i, val in enumerate(values):
            arms_utility_hindsight = np.zeros(len(arms))
            
            for j, arm in enumerate(arms):
                if(self.auction_type == 'SecondPrice'):
                    # val - prices[i] -> since if i exceed prices[i] i'd still pay prices[i] (the 2nd price)
                    arms_utility_hindsight[j] = val - prices[i] if arm >= prices[i] else 0
                elif(self.auction_type == 'FirstPrice'):
                    # val - arm       -> since if i exceed prices[i] i'd now pay my own arm (the 1st price)
                    arms_utility_hindsight[j] = val - arm if arm >= prices[i] else 0

            # calculate pivotal bid:
            # 1) get the max utility mask
            # 2) pivotal varies with the max utility
            #   if  > 0: pivotal is min bid that achieves it
            #       (only useful in 2nd price -> every bid above pivotal has same price, hence same utility) 
            #   if == 0: pivotal is max bid that doesn't lose money
            pivotal_bid = arms[arms_utility_hindsight==arms_utility_hindsight.max()]
            pivotal_bid = pivotal_bid.min() if arms_utility_hindsight.max()>0 else pivotal_bid.max()
            rewards_hindsight[i] = (pivotal_bid, arms_utility_hindsight.max())

        regrets_hindsight = rewards_hindsight[:, 1] - surpluses
        return rewards_hindsight, regrets_hindsight
    

################################
######  Truthful Bandit   ######
################################
class TruthfulBandit(BaseBandit):
    def __init__(self, rng):
        super(TruthfulBandit, self).__init__(rng)
        self.truthful = True

    def bid(self, value, context, estimated_CTR):
        return value*estimated_CTR
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        self.regret.append(0)   # truthful is no-regret


################################
######        UCB-1       ######
################################
from math import sqrt, log, log10

class UCB1(BaseBandit):
    def __init__(self, rng, gamma=1):
        super(UCB1, self).__init__(rng)
        self.gamma = gamma

        # self.total_reward = 0
        self.payoffs = [[] for _ in range(self.NUM_BIDS)]

        self.num_auctions = 0
        self.ucbs = [float('inf')] * self.NUM_BIDS
        self.played_arms = []

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize,
               fontsize, name):
        # payoff as valuations
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]/2

        # Update payoffs and total payoff
        for bid in self.BIDS:
            played_bid_mask = np.array([(played_arm == bid) for played_arm in self.played_arms])
            bid_utilities = utilities[played_bid_mask]
            i = self.BIDS.index(bid)
            self.payoffs[i].append(bid_utilities.sum())
            self.expected_utilities[i] = np.array(self.payoffs[i]).mean()
        self.total_reward += utilities.sum()

        # Calculate UCB for each bid
        for i in range(self.NUM_BIDS):
            if self.counters[i] == 0:
                # If bid has not been made before, set UCB to infinity
                self.ucbs[i] = float('inf')
            else:
                # Calculate UCB using UCB-1 formula
                self.ucbs[i] = self.expected_utilities[i] + self.gamma *\
                               sqrt(2*log(self.num_auctions / self.counters[i]))
                
        # IN HINDSIGHT
        action_rewards, regrets = self.calculate_regret_in_hindsight(self.BIDS, values, prices, utilities)
        self.regret.append(regrets.sum())  # sum over rounds_per_iter=10 auctions
        self.actions_rewards.append(action_rewards)

    def bid(self, value, context, estimated_CTR):
        self.num_auctions += 1
        # 1/sqrtN -> 1, 1/sqrt2, 1/sqrt3 ... (decaying slower than 1/n)
        # at n=1000 you have ~3%, instead of 0.1% given by 1/n

        max_ucb = max(self.ucbs)
        max_ucbs_mask = [(ucb==max_ucb) for ucb in self.ucbs]
        bids_w_max_ucb = [bid for bid in self.BIDS if max_ucbs_mask[self.BIDS.index(bid)]]
        chosen_bid = self.rng.choice(bids_w_max_ucb)

        self.counters[self.BIDS.index(chosen_bid)] += 1
        self.played_arms.append(chosen_bid)
        return chosen_bid

    # def bid(self, value, context, estimated_CTR):   # -> truthful bidding?
    #     return value * estimated_CTR

    def clear_logs(self, memory):
        self.played_arms = []


################################
######   EPSILON-GREEDY   ######
################################
class EpsilonGreedy(BaseBandit):
    def __init__(self, rng):
        super(EpsilonGreedy, self).__init__(rng)

        # Actions -> discrete, finite, fixed
        # self.BIDS = [0.005, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.9, 2.4]
        # self.NUM_BIDS = len(self.BIDS)
        # self.counters = [0] * self.NUM_BIDS

        # Reward -> avg payoff history
        # self.expected_utilities = [0] * self.NUM_BIDS

        # self.total_reward = 0
        self.history_payoffs = [[] for _ in range(self.NUM_BIDS)]
        self.num_auctions = 0
        self.played_arms = []

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # Update payoffs
        # Update Expected Utility
        for i,bid in enumerate(self.BIDS):
            played_bid_mask = np.array([(played_arm == bid) for played_arm in self.played_arms])
            bid_utilities = utilities[played_bid_mask]
            bid_payoffs = self.history_payoffs[i]
            bid_payoffs.append( bid_utilities.sum() )
            self.expected_utilities[i] = np.array(self.history_payoffs[i]).mean()
        self.total_reward += utilities.sum()
        
        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight(self.BIDS, values, prices, utilities)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)

    def bid(self, value, context, estimated_CTR):
        self.num_auctions += 1
        # random exploration grows as 1/sqrt(n)

        if self.rng.random() <= (1 / sqrt(self.num_auctions)):
            chosen_bid = self.rng.choice(self.BIDS)
        else:
            max_utility = max(self.expected_utilities)
            max_utilities_mask = [(u == max_utility) for u in self.expected_utilities]
            best_bids = [bid for bid in self.BIDS if max_utilities_mask[self.BIDS.index(bid)]]
            chosen_bid = self.rng.choice(best_bids)

        self.counters[self.BIDS.index(chosen_bid)] += 1
        self.played_arms.append(chosen_bid)
        return chosen_bid

    def clear_logs(self, memory):
        self.played_arms = []


################################
######        Exp3        ######
################################
class Exp3(BaseBandit):
    def __init__(self, rng, gamma=1):
        super(Exp3, self).__init__(rng)
        self.gamma = gamma

        self.max_weight = 1e4

        self.expected_utilities = np.zeros(self.NUM_BIDS)
        self.counters = np.zeros(self.NUM_BIDS)
        self.w = np.ones(self.NUM_BIDS)
        self.p = np.ones(self.NUM_BIDS) / self.NUM_BIDS
        self.p[0] = 1 - self.p[1:].sum()    # make sure that sum(p)=1 
        self.p_history = []

        # regret
        self.actions_rewards = []       # not used for now

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        # qua sotto | per avereoutcomes sempre 1!
        # surpluses[won_mask] = values[won_mask] - prices[won_mask]

        # IN HINDSIGHT
        action_rewards, regrets = self.calculate_regret_in_hindsight(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())  # sum over rounds_per_iter=10 auctions
        self.actions_rewards.append(action_rewards)

        pass
        for i, arm in enumerate(self.BIDS):
            arm_mask = np.array([(played_arm == arm) for played_arm in bids])
            
            if surpluses[arm_mask].size > 0:
                self.counters[i] += surpluses[arm_mask].size
                self.expected_utilities[i] = (
                    self.expected_utilities[i] * self.counters[i] +
                    surpluses[arm_mask].mean() * surpluses[arm_mask].size ) \
                    / self.counters[i]
            arm_surpluses = surpluses[arm_mask & won_mask]
            # weight update of exp3 substituted with exp(arctan())
            delta_w = np.array([ np.exp(np.arctan(self.gamma * r / self.p[i]))  for r in arm_surpluses]).prod()
            
            candidate_w = self.w[i]*delta_w
            if candidate_w < 0: 
                candidate_w = self.max_weight
            self.w[i] = min(candidate_w, self.max_weight)     # clip to avoid overflow
            # for reward in arm_surpluses:
            #     weight_update = np.exp(self.gamma * reward / self.p[i])
            #     self.w[i] *= weight_update
        
        self.p_history.append(self.p.copy())
        self.p = self.w / self.w.sum()
        self.p[0] = 1 - self.p[1:].sum()
        if (self.p<0).any():
            raise ValueError("Negative probability in Exp3", self.p)

    def bid(self, value, context, estimated_CTR):
        pass
        pulled_arm = self.rng.choice(self.NUM_BIDS, p=self.p)
        return self.BIDS[pulled_arm]

    def clear_logs(self, memory):
        #   TODO
        #   needed????
        pass
