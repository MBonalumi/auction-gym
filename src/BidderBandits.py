import numpy as np
from Bidder import Bidder
import matplotlib.pyplot as plt


################################
######     Base Bandit    ######
################################
class BaseBandit(Bidder):
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions"):
        super(BaseBandit, self).__init__(rng)
        self.agent_id = -1
        self.auction_type = "SecondPrice"
        self.num_iterations = -1

        # Actions -> discrete, finite, fixed
        self.isContinuous = isContinuous
        self.textContinuous = textContinuous
        # self.BIDS = np.array([0.005, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.9, 2.4])
        self.BIDS = np.array([0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.1, 1.4])
        self.NUM_BIDS = self.BIDS.size
        self.counters = np.zeros_like(self.BIDS)

        # Reward -> avg payoff history
        self.expected_utilities = np.zeros_like(self.BIDS)

        self.regret = []
        self.actions_rewards = []       # not used for now
        self.total_reward = 0
        self.total_regret = 0

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]
                
        # IN HINDISGHT
        if self.isContinuous:
            actions_rewards, regrets = self.calculate_regret_in_hindsight_continuous(bids, values, prices, surpluses)
        else:
            actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!
        return

    def bid(self, value, context, estimated_CTR):
        return 0.123456789      # recognizable value
    
    def clear_logs(self, memory):
        pass

    def calculate_regret_in_hindsight_continuous(self, bids, values, prices, surpluses):
        '''
        function that calculates
        rewards and regrets in hindsight
        for a batch of auctions'
        in case the agent draws actions from a continuous distribution
        '''
        # TODO: I DONT HAVE MAX PRICE! CANT DECIDE WIN IN 2ND PRICE AUCTION

        actions_rewards_in_hs = np.zeros((values.size, 2))      # tuples (arm, reward)

        for i in range(len(values)):

            if self.auction_type == 'SecondPrice':
                win_bid_in_hs = prices[i] + 0.01    # NOT TRUE! PRICES IS NOT WINNING BID!
                utility_in_hs = max(0, values[i]-win_bid_in_hs)
                best_bid_in_hs = win_bid_in_hs if utility_in_hs > 0 else prices[i] - 0.01
                actions_rewards_in_hs[i] = (best_bid_in_hs, utility_in_hs)

            elif self.auction_type == 'FirstPrice':
                win_bid_in_hs = prices[i] + 0.01
                utility_in_hs = max(0, values[i]-win_bid_in_hs)
                best_bid_in_hs = win_bid_in_hs if utility_in_hs > 0 else prices[i] - 0.01
                actions_rewards_in_hs[i] = (best_bid_in_hs, utility_in_hs)

        regrets_in_hs = actions_rewards_in_hs[:, 1] - surpluses
        return actions_rewards_in_hs, regrets_in_hs


    def calculate_regret_in_hindsight_discrete(self, arms, values, prices, surpluses):
        '''
        function that calculates
        rewards and regrets in hindsight
        for a batch of auctions
        '''
        # TODO: I DONT HAVE MAX PRICE! CANT DECIDE WIN IN 2ND PRICE AUCTION

        actions_rewards_in_hindsight = np.zeros((values.size, 2))      # tuples (arm, reward)
        
        for i, val in enumerate(values):
            arms_utility_in_hindsight = np.zeros(len(arms))
            
            for j, arm in enumerate(arms):
                if(self.auction_type == 'SecondPrice'):
                    # val - prices[i] -> since if i exceed prices[i] i'd still pay prices[i] (the 2nd price)
                    ''' BUT I DON'T KNOW THE WINNING BID!!! SO I CAN'T SAY WHICH ARMS WOULD HAVE WON '''
                    arms_utility_in_hindsight[j] = val - prices[i]     if arm >= prices[i]  else 0
                elif(self.auction_type == 'FirstPrice'):
                    # val - arm       -> since if i exceed prices[i] i'd now pay my own arm (the 1st price)
                    arms_utility_in_hindsight[j] = val - arm     if arm >= prices[i]  else 0

            # calculate pivotal bid:
            # 1) get the max utility mask
            # 2) pivotal varies with the max utility
            #   if  > 0: pivotal is closest 0-utility bid (min) to positive utilities
            #       (only useful in 2nd price -> every bid above pivotal has same price, hence same utility) 
            #   if == 0: pivotal is closest 0-utility bid (max) to negative utilities
            pivotal_bid = arms[arms_utility_in_hindsight==arms_utility_in_hindsight.max()]
            pivotal_bid = pivotal_bid.min() if arms_utility_in_hindsight.max()>0 else pivotal_bid.max()
            actions_rewards_in_hindsight[i] = (pivotal_bid, arms_utility_in_hindsight.max())

        regrets_in_hindsight = actions_rewards_in_hindsight[:, 1] - surpluses
        return actions_rewards_in_hindsight, regrets_in_hindsight
    

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

class TruthfulBandit_gather_data(TruthfulBandit):
    def __init__(self, rng):
        super(TruthfulBandit_gather_data, self).__init__(rng)
        self.contexts = []
        self.values = []
        self.bids = []
        self.prices = []
        self.outcomes = []
        self.estimated_CTRs = []
        self.won_mask = []

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        self.contexts.extend(contexts)
        self.values.extend(values)
        self.bids.extend(bids)
        self.prices.extend(prices)
        self.outcomes.extend(outcomes)
        self.estimated_CTRs.extend(estimated_CTRs)
        self.won_mask.extend(won_mask)

        if iteration == self.num_iterations-1:
            self.contexts = np.array(self.contexts)
            self.values = np.array(self.values)
            self.bids = np.array(self.bids)
            self.prices = np.array(self.prices)
            self.outcomes = np.array(self.outcomes)
            self.estimated_CTRs = np.array(self.estimated_CTRs)
            self.won_mask = np.array(self.won_mask)

            np.savez_compressed('data/10mln_data_samples.npz', 
                contexts=self.contexts, values=self.values, bids=self.bids, prices=self.prices, 
                outcomes=self.outcomes, estimated_CTRs=self.estimated_CTRs, won_mask=self.won_mask)




################################
######        UCB-1       ######
################################

# https://github.com/marcomussi/DLB

# from math import sqrt, log, log10

class UCB1(BaseBandit):
    def __init__(self, rng, sigma=1):
        super(UCB1, self).__init__(rng)
        self.sigma = sigma

        # self.total_reward = 0

        self.t = 0
        self.ucbs = [float('inf')] * self.NUM_BIDS

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        self.t += values.size
        
        # SURPLUSES
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # IN HINDSIGHT
        action_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())  # sum over rounds_per_iter=10 auctions
        self.actions_rewards.append(action_rewards)

        # Update payoffs and total payoff
        played_bids = set(bids)
        for bid in played_bids:
            # handles all auctions where bid was played
            # only for those bids that have been played
            mask = bids == bid
            bid_utilities = surpluses[mask]
            n_plays = bid_utilities.size
            i = (self.BIDS == bid).nonzero()[0][0]
            self.expected_utilities[i] = (self.expected_utilities[i] * self.counters[i] + bid_utilities.sum()) / (self.counters[i] + n_plays)
            self.counters[i] += n_plays

            '''no bid inside this for should have counter == 0'''
            # #update UCB
            # if self.counters[i] == 0:
            #     # If bid has not been made before, set UCB to infinity
            #     self.ucbs[i] = float('inf')
            
            # Calculate UCB using UCB-1 formula
            self.ucbs[i] = self.expected_utilities[i] + self.sigma * np.sqrt(2 * np.log(self.t) / self.counters[i])
            
        self.total_reward += surpluses.sum()

    def bid(self, value, context, estimated_CTR):
        # 1/sqrtN -> 1, 1/sqrt2, 1/sqrt3 ... (decaying slower than 1/n)
        # at n=1000 you have ~3%, instead of 0.1% given by 1/n

        max_ucb = max(self.ucbs)
        max_ucbs_mask = [(ucb==max_ucb) for ucb in self.ucbs]
        bids_w_max_ucb = self.BIDS[max_ucbs_mask]
        chosen_bid = self.rng.choice(bids_w_max_ucb)

        return chosen_bid


################################
######   EPSILON-GREEDY   ######
################################
class EpsilonGreedy(BaseBandit):
    def __init__(self, rng):
        super(EpsilonGreedy, self).__init__(rng)
        self.t = 0

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # SURPLUSES
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)

        # Update Expected Utility
        played_bids = set(bids)
        for bid in played_bids:
            mask = bids == bid
            bid_utilities = surpluses[mask]
            n_plays = bid_utilities.size
            i = (self.BIDS == bid).nonzero()[0][0]
            self.expected_utilities[i] = (self.expected_utilities[i] * self.counters[i] + bid_utilities.sum()) / (self.counters[i] + n_plays)
            self.counters[i] += n_plays
        self.total_reward += surpluses.sum()
        
    def bid(self, value, context, estimated_CTR):
        self.t += 1
        # random exploration grows as 1/sqrt(n)

        if self.rng.random() <= (1 / np.sqrt(self.t)):
            chosen_bid = self.rng.choice(self.BIDS)
        else:
            max_utility = max(self.expected_utilities)
            max_utilities_mask = [(u == max_utility) for u in self.expected_utilities]
            best_bids = self.BIDS[max_utilities_mask]
            chosen_bid = self.rng.choice(best_bids)

        return chosen_bid


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

        # DONE BY super().update()
        # # IN HINDSIGHT
        # action_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        # self.regret.append(regrets.sum())  # sum over rounds_per_iter=10 auctions
        # self.actions_rewards.append(action_rewards)

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

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


################################
#####       GP_UCB        ######
################################

### suorce: https://github.com/tushuhei/gpucb

from sklearn.gaussian_process import GaussianProcessRegressor
class gp_ucb(BaseBandit):
    def __init__(self, rng, beta=100, arms_amount=20):
        super(gp_ucb, self).__init__(rng)
        self.BIDS = np.array(range(5, 3000, (int(2995/arms_amount))+1)) / 1000
        self.NUM_BIDS = len(self.BIDS)
        self.beta = beta        # exploration hyperparam
        self.mu = np.array([0. for _ in range(self.NUM_BIDS)])          # ucb mean 
        self.sigma = np.array([0.5 for _ in range(self.NUM_BIDS)])      # ucb stddev
        self.X = []     # bids history
        self.Y = []     # surpluses history

    def bid(self, value, context, estimated_CTR):
        idx = np.argmax(self.mu + self.sigma * np.sqrt(self.beta))
        chosen_bid = self.BIDS[idx]
        return chosen_bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]

        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # not batched!!!

        # GP Regression
        '''
        quello che calcolo non ha senso
        '''
        self.X.append(bids)
        self.Y.append(surpluses)
        x = np.array(self.X).reshape(-1,1)[-self.learning_window:]
        y = np.array(self.Y).reshape(-1,1).squeeze()[-self.learning_window:]
        pass
        # assert x.shape[0] == y.shape[0]
        gp = GaussianProcessRegressor()
        gp.fit(x, y)
        # data = np.array(self.BIDS).reshape(1, -1)
        self.mu, self.sigma = gp.predict(np.array(self.BIDS).reshape(-1,1), return_std=True)    # why self.BIDS???



##########################################
#####    warm-start GPRegression    ######
##########################################
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

class warm_start_gpr(BaseBandit):
    def __init__(self, rng):
        super(warm_start_gpr, self).__init__(rng)
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        self.random_state = rng.choice(100)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, random_state=self.random_state)

        # self.X = []    # bids history (TO BE CLEARED AFTER UPDATE)
        # self.y = []    # surpluses history (TO BE CLEARED AFTER UPDATE)
    
    def bid(self, value, context, estimated_CTR):
        expected_rewards = np.zeros_like(self.BIDS)
        for i, bid in enumerate(self.BIDS):
            expected_rewards[i] = self.gpr.predict(bid.reshape(1,-1))
        chosen_bid = self.BIDS[np.argmax(expected_rewards)]

        return chosen_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]

        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # not batched!!!

        # GP Regression
        X = bids.reshape(-1,1)
        y = surpluses.reshape(-1,1)
        gp = GaussianProcessRegressor(kernel=self.kernel, random_state=self.random_state).fit(X, y)

        self.kernel.set_params(**(gp.kernel_.get_params()))
        self.gpr = gp

    def clear_logs(self, memory):
        pass
        # del self.X
        # del self.y
        # self.X = []
        # self.y = []


################################
#####  incremental GPReg  ######
################################

### source: https://github.com/Bigpig4396/Incremental-Gaussian-Process-Regression-IGPR

from ModelsMine import IGPR
class IGPRBidder(BaseBandit):
    def __init__(self, rng, arms_amount=20):
        super(IGPRBidder, self).__init__(rng)
        #self.BIDS = np.array(range(5, 3000, (int(2995/arms_amount))+1)) / 1000
        self.NUM_BIDS = len(self.BIDS)
        self.igpr = IGPR(init_x=0, init_y=0)

        # self.X = []     # bids history
        # self.Y = []     # surpluses history

        self.fit_once = False

    def bid(self, value, context, estimated_CTR):
        if self.fit_once:
            expected_rewards = np.zeros_like(self.BIDS)
            for i, bid in enumerate(self.BIDS):
                expected_rewards[i] = self.igpr.predict(bid)
            chosen_bid = self.BIDS[np.argmax(expected_rewards)]
        else:
            chosen_bid = self.rng.choice(self.BIDS)
        return chosen_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]
                
        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(self.BIDS, values, prices, surpluses)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!

        # partial_fit
        # self.X.append(bids)
        # self.Y.append(surpluses)
        pass
        # for x, y in zip(self.X, self.Y):
        self.igpr.learn(new_x=bids, new_y=surpluses)
