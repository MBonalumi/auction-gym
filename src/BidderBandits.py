import numpy as np
from Bidder import Bidder
import matplotlib.pyplot as plt
from numba import jit


################################
######     Base Bandit    ######
################################
class BaseBidder(Bidder):
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions", save_model=False):
        super(BaseBidder, self).__init__(rng)
        self.agent_id = -1
        self.auction_type = "SecondPrice"
        self.num_iterations = -1
        self.total_num_auctions = -1

        self.item_values = None

        # Actions -> discrete, finite, fixed
        self.isContinuous = isContinuous
        self.textContinuous = textContinuous
        # self.BIDS = np.array([0.005, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.9, 2.4])
        # self.BIDS = np.array([0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.1, 1.4], dtype=np.float32)
        # self.BIDS = np.array([0.1, 0.5, 1.0], dtype=np.float32)
        self.BIDS = np.array([0.1, 0.3, 0.5, 0.7, 1.0], dtype=np.float32)
        self.NUM_BIDS = self.BIDS.size
        self.counters = np.zeros_like(self.BIDS)

        # Reward -> avg payoff history
        self.expected_utilities = np.zeros_like(self.BIDS)

        self.winning_bids = np.zeros(1)     # winning bids for each iteration, set manually here
                                            # used to calculate regret in hindsight (2nd price)
        self.second_winning_bids = np.zeros(1)     # 2nd winning bids, used to calculate regret in hindsight (1st price)
        self.regret = []
        self.surpluses = []
        self.expected_surpluses = []
        self.actions_rewards = []       # not used for now
        self.total_reward = 0
        self.total_regret = 0

        self.save_model = save_model

        self.clairevoyant = None
        self.clairevoyant_regret = []
        self.ctrs = []      #save only for bidders with a clairevoyant

        self.bids = []

        self.contexts = []

        from BidderNovelty import NoveltyClairevoyant
        self.is_clairevoyant = isinstance(self, NoveltyClairevoyant)
        self.arms_utility_in_hindsight = []

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        assert self.winning_bids.size == bids.size, "ERROR: winning_bids.size != bids.size"
        assert self.second_winning_bids.size == bids.size, "ERROR: 2nd winning_bids.size != bids.size"

        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]

        expected_surpluses = np.zeros_like(values)
        expected_surpluses[won_mask] = values[won_mask]  * estimated_CTRs[won_mask] - prices[won_mask]

        # IN HINDISGHT
        if self.isContinuous:
            actions_rewards, regrets = self.calculate_regret_in_hindsight_continuous(bids, values, prices, expected_surpluses, estimated_CTRs, outcomes)
        else:
            actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, expected_surpluses, estimated_CTRs, outcomes)
        # self.regret.append(regrets.sum())
        self.regret.extend(regrets)     # batch not averaged !!!
        self.surpluses.extend(surpluses)    # batch not averaged !!!
        self.expected_surpluses.extend(expected_surpluses)    # batch not averaged !!!
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!
        self.contexts.extend(contexts)
        self.bids.extend(bids)
        
        # Compute CV Regret
        if self.clairevoyant is not None:
            self.clairevoyant_regret.extend(self.compute_cv_regret(contexts, expected_surpluses, values, bids, estimated_CTRs))
            self.ctrs.extend(estimated_CTRs)
        
        return actions_rewards, regrets

    def bid(self, value, context, estimated_CTR):
        return 0.123456789      # recognizable value
    
    def clear_logs(self, memory):
        pass

    def calculate_regret_in_hindsight_continuous(self, bids, values, prices, expected_surpluses, estimated_CTRs, outcomes):
        '''
        function that calculates
        rewards and regrets in hindsight
        for a batch of auctions'
        in case the agent draws actions from a continuous distribution
        '''

        actions_rewards_in_hs = np.zeros((values.size, 2))      # tuples (arm, reward)

        for i in range(len(values)):
            # bid_to_beat is the bid that would have won the auction, excluding mine since i am recalculating it
            bid_to_beat = self.winning_bids[i] if self.winning_bids[i] != bids[i] else self.second_winning_bids[i]

            win_bid_in_hs = bid_to_beat + 0.01
            price_in_hs_if_win = win_bid_in_hs if self.auction_type == 'FirstPrice' else bid_to_beat # SecondPrice
            utility_in_hs = max( 0 ,  values[i] - price_in_hs_if_win )
            best_bid_in_hs = win_bid_in_hs if utility_in_hs > 0 else values[i]
            actions_rewards_in_hs[i] = (best_bid_in_hs, utility_in_hs)

        regrets_in_hs = actions_rewards_in_hs[:, 1] - expected_surpluses
        return actions_rewards_in_hs, regrets_in_hs

    def calculate_regret_in_hindsight_discrete(self, bids, values, prices, expected_surpluses, estimated_CTRs, outcomes):
        '''
        function that calculates
        rewards and regrets in hindsight
        for a batch of auctions
        '''
        actions_rewards = np.zeros((values.size, 2))      # tuples (arm, reward)
        
        for i, val in enumerate(values):
            arms_utility_in_hindsight = np.zeros(len(self.BIDS))
            bid_to_beat = self.winning_bids[i] if self.winning_bids[i] != bids[i] else self.second_winning_bids[i]        # winning bid if it's not mine, else second
            ctr = estimated_CTRs[i] if estimated_CTRs is not None else 1.0
            # buys = outcomes[i] if outcomes is not None else True

            if(self.auction_type == 'SecondPrice'):
                # val - prices[i] -> since if i exceed prices[i] i'd still pay prices[i] (the 2nd price)
                utility = lambda    arm:    val * ctr - bid_to_beat     if arm >= bid_to_beat     else 0
            elif(self.auction_type == 'FirstPrice'):
                # val - arm       -> since if i exceed prices[i] i'd now pay my own arm (the 1st price)
                utility = lambda    arm:    val * ctr  - arm             if arm >= bid_to_beat     else 0

            for j, arm in enumerate(self.BIDS):
                arms_utility_in_hindsight[j] = utility(arm)


            # calculate pivotal bid:
            # 1) get the max utility mask
            # 2) pivotal varies with the max utility
            #   if  > 0: pivotal is closest 0-utility bid (min) to positive utilities
            #       (only useful in 2nd price -> every bid above pivotal has same price, hence same utility) 
            #   if == 0: pivotal is closest 0-utility bid (max) to negative utilities
            pivotal_bid = self.BIDS[arms_utility_in_hindsight==arms_utility_in_hindsight.max()]
            # pivotal_bid = pivotal_bid.min() if arms_utility_in_hindsight.max()>0 else pivotal_bid.max()
                # the calc above is wrong, i want to bid 0.0 if i have no utility, not the closest bid to the auction win
            pivotal_bid = pivotal_bid.min() 
            actions_rewards[i] = (pivotal_bid, arms_utility_in_hindsight.max())

            if self.is_clairevoyant:
                self.arms_utility_in_hindsight.append(arms_utility_in_hindsight)

        regrets = actions_rewards[:, 1] - expected_surpluses
        return actions_rewards, regrets
    
    def compute_cv_regret(self, contexts, expected_surpluses, values, bids, estimated_CTRs):
        mkt_prices = self.clairevoyant.predict(contexts).squeeze()

        mask_winnable = np.max(self.BIDS) > mkt_prices    # at least one bid > mkt_price
        optimal_bids = np.zeros_like(mkt_prices)
        optimal_bids[mask_winnable] = np.array(
            [ np.min(self.BIDS[self.BIDS-mkt_price > 0.0])   for mkt_price in mkt_prices[mask_winnable] ]
        )
        optimal_bids[optimal_bids > values] = 0.0   # for sure not profitable

        # calculate surpluses based on the actual market price?
        # YES
        real_mkt_prices = self.winning_bids
        real_mkt_prices[real_mkt_prices == bids] = self.second_winning_bids[real_mkt_prices == bids]
        
        prices = optimal_bids if self.auction_type == 'FirstPrice' else real_mkt_prices # SecondPrice
        cv_surpluses = (optimal_bids > real_mkt_prices) * (values * estimated_CTRs - prices)
        cv_regrets = cv_surpluses - expected_surpluses
        return cv_regrets
    

################################
#####    STATIC BIDDERS    #####
################################
class StaticBidder(BaseBidder):
    '''
    Used to distinguish Static and non Static bidders

    for NON-static bidders we compute the regret with respect to a "on-average"-optimal clairevoyant 
    '''
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions", save_model=False):
        super().__init__(rng, isContinuous, textContinuous, save_model)


#################################
######   static π Bidder   ######
#################################
from math import erf
class StaticBidder1(StaticBidder):
    def __init__(self, rng, bid_interval=(0, 1), bid_prob_weights=(1., 1., 1., 1., 1., 1.), bid_prob_tendency=0.5):
        super(StaticBidder1, self).__init__(rng)
        self.static = True
        self.bid_interval = bid_interval      # (min, max)
        self.bid_prob_weights = bid_prob_weights   # (0,1) for each dimension of context
        self.bid_prob_tendency = bid_prob_tendency

        # used to normalize bid_prob @ context
        self.ctxt_var = 1.0      # gym should set the actual value
        self.ctxt_mean = 0.0     # gym should set the actual value


    def bid(self, value, context, estimated_CTR):
        zscore = ( (self.bid_prob_weights @ context) - self.ctxt_mean) / np.sqrt(self.ctxt_var)
        prob = 0.5 * (1 + erf( zscore / np.sqrt(2) )) * self.bid_prob_tendency
        if self.rng.random() < prob:
            return self.rng.uniform(self.bid_interval[0], self.bid_interval[1])
        else:
            return 0.0


#################################
#####   static π Bidder 2   #####
#################################
# import math

# def exponential(x):
#     return math.exp(x)

# Calculate e^x using sum of first n terms of Taylor Series
@jit(nopython=True)
def exponential(x, n=10):
    # initialize sum of series
    sum = 1.0
    for i in range(n, 0, -1):
        sum = 1 + x * sum / i
    return sum

@jit(nopython=True)
def inverse_logit(x):
    exp_x = exponential(x)
    return exp_x / (1 + exp_x)

class StaticBidder2(StaticBidder):
    def __init__(self, rng, bid_prob_weights=(.2, .2, .2, .2, .2, 0.), noise_variance=0.02 ):
        super(StaticBidder2, self).__init__(rng)
        self.isContinuous = True
        self.static = True
        self.bid_prob_weights = bid_prob_weights   # in the simplex, meaning weights.sum()==1.0
        self.noise_variance = noise_variance

        self.to_logitnormal = lambda exp_x:  exp_x / (1 + exp_x)

    def bid(self, value, context, estimated_CTR):
        logit_context = np.array([inverse_logit(c) for c in context])
        bid = (logit_context @ self.bid_prob_weights) * value
        bid += self.rng.normal(0, self.noise_variance * value)
        bid = np.maximum(0, bid)
        # del logit_context

        discretized_bid = self.BIDS[np.argmin(np.abs(self.BIDS - bid))]

        return discretized_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)


###################################
######   static π Bidder 2   ######
###################################

### SMALLER CONTEXT, USES ONLY FIRST DIMENSION

class StaticBidder2_SmallContext(StaticBidder2):
    def __init__(self, rng, bid_prob_weights=(0.2, 0.2, 0.2, 0.2, 0.2, 0), noise_variance=0.02):
        super().__init__(rng, bid_prob_weights, noise_variance)
        self.noise_variance = noise_variance
        # self.interest = self.rng.uniform(-1, 1)

    def bid(self, value, context, estimated_CTR):
        '''
        context[0] can be -1.09, 0.0, 1.09
        bid(-1.09) = 0.298
        bid(  0.0) = 0.593
        bid( 1.09) = 0.887
        '''
        logit_context = inverse_logit(context[0])
        bid = logit_context * value
        bid += self.rng.normal(0, self.noise_variance * value)
        bid = np.maximum(0., bid)
        del logit_context
        return bid

################################
######  Truthful Bandit   ######
################################
class TruthfulBandit(BaseBidder):
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

            np.savez_compressed('data/10mln_data_samples_NEW.npz', 
                contexts=self.contexts, values=self.values, bids=self.bids, prices=self.prices, 
                outcomes=self.outcomes, estimated_CTRs=self.estimated_CTRs, won_mask=self.won_mask)




################################
######        UCB-1       ######
################################

# https://github.com/marcomussi/DLB

# from math import sqrt, log, log10

class UCB1(BaseBidder):
    def __init__(self, rng, sigma=1):
        super(UCB1, self).__init__(rng)
        self.sigma = sigma

        # self.total_reward = 0

        self.t = 0
        self.ucbs = np.array([float('inf')] * self.NUM_BIDS)

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        self.t += values.size
        
        # SURPLUSES
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

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

        # max_ucb = max(self.ucbs)
        # max_ucbs_mask = [(ucb==max_ucb) for ucb in self.ucbs]
        # bids_w_max_ucb = self.BIDS[max_ucbs_mask]
        # chosen_bid = self.rng.choice(bids_w_max_ucb)

        max_ucb_bids = self.BIDS[self.ucbs == self.ucbs.max()]
        chosen_bid = self.rng.choice(max_ucb_bids)
        return chosen_bid


################################
######   EPSILON-GREEDY   ######
################################
class EpsilonGreedy(BaseBidder):
    def __init__(self, rng):
        super(EpsilonGreedy, self).__init__(rng)
        self.t = 0

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # SURPLUSES
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses, estimated_CTRs)
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
np.warnings.filterwarnings('ignore', category=RuntimeWarning)

class Exp3(BaseBidder):
    def __init__(self, rng, learning_rate=1):
        super(Exp3, self).__init__(rng)
        '''
        gamma = min(1, cubic_root( (K * ln K)/(2 * g) ))
            with K being the number of arms
            with g being an upper bound of the sum of all rewards the best algorithm can get
                in my case  g = value * total_num_auctions
        '''
        self.learning_rate = 0.05   # gamma = cubic_root( (11 * ln11)/(2 * 118'000) )

        # self.max_weight = 1e6

        self.t = 0

        self.expected_utilities = np.zeros(self.NUM_BIDS)
        self.counters = np.zeros(self.NUM_BIDS)
        self.w = np.ones(self.NUM_BIDS)
        self.p = np.ones(self.NUM_BIDS, dtype=np.float64) / self.NUM_BIDS
        self.p[0] = 1 - self.p[1:].sum()    # make sure that sum(p)=1 
        self.p_history = []

        # regret
        self.actions_rewards = []       # not used for now

    # def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
    #     surpluses = np.zeros_like(values)
    #     surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

    #     super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

    #     for i, arm in enumerate(self.BIDS):
    #         arm_mask = np.array([(played_arm == arm) for played_arm in bids])
            
    #         if surpluses[arm_mask].size > 0:
    #             self.counters[i] += surpluses[arm_mask].size
    #             self.expected_utilities[i] = (
    #                 self.expected_utilities[i] * self.counters[i] +
    #                 surpluses[arm_mask].mean() * surpluses[arm_mask].size ) \
    #                 / self.counters[i]
    #         arm_surpluses = surpluses[arm_mask & won_mask]
    #         # weight update of exp3 substituted with exp(arctan())
    #         delta_w = np.array([ np.exp(np.arctan(self.learning_rate * r / self.p[i]))  for r in arm_surpluses]).prod()
            
    #         candidate_w = self.w[i]*delta_w
    #         if candidate_w < 0: 
    #             candidate_w = self.max_weight
    #         # self.w[i] = min(candidate_w, self.max_weight)     # clip to avoid overflow
    #         self.w[~np.isfinite(self.w)] = 0    # disactivate arms with infinite weight 
        
    #     self.p_history.append(self.p.copy())
    #     self.p = self.w / self.w.sum()
    #     self.p[0] = 1 - self.p[1:].sum()
    #     if (self.p<0).any():
    #         raise ValueError("Negative probability in Exp3", self.p)

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        rewards = surpluses / values       # rewards are normalized, in [0,1]

        # for each bid(arm) played, update the expected reward for that bid(arm)
        for i, bid in enumerate(bids):
            arm_id = np.where(self.BIDS == bid)[0][0]
            self.expected_utilities[arm_id] += rewards[i] / self.p[arm_id]
            self.w[arm_id] = np.exp(self.learning_rate * self.expected_utilities[arm_id] / self.NUM_BIDS)
            # self.w[arm_id] = min(self.w[arm_id], self.max_weight)     # clip to avoid overflow
            self.w[~np.isfinite(self.w)] = 0    # disactivate arms with infinite weight
            # self.p = self.w / self.w.sum()
            self.p = (1 - self.learning_rate) * self.w / self.w.sum()  +  self.learning_rate / self.NUM_BIDS / self.t
        
        self.p = self.p / self.p.sum()
        self.p[0] = 1 - self.p[1:].sum()

        if (self.p<0).any():
            raise ValueError("Negative probability in Exp3: ", self.p)

    def bid(self, value, context, estimated_CTR):
        self.t += 1
        if (self.p<0).any():
            raise ValueError("Negative probability in Exp3: ", self.p)
        if  np.abs(self.p.sum() - 1) > 1e-6:
            raise ValueError("dont sum to 1: ", self.p)
        pulled_arm = self.rng.choice(self.NUM_BIDS, p=self.p)
        return self.BIDS[pulled_arm]


################################
#####    Exp3 Gianmarco    #####
################################
class Exp3Gianmarco(BaseBidder):
    def __init__(self, rng, learning_rate=1):
        super(Exp3Gianmarco, self).__init__(rng)
        '''
        learning_rate = min(1, cubic_root( (K * ln K)/(2 * g) ))
            with K being the number of arms
            with g being an upper bound of the sum of all rewards the best algorithm can get
                in my case  g = value * total_num_auctions
        '''
        self.gamma = 0.05   # learning_rate = cubic_root( (11 * ln11)/(2 * 118'000) )

        self.w = np.ones(self.NUM_BIDS)
        self.est_rewards = np.zeros(self.NUM_BIDS)
        self.probabilities = (1/self.NUM_BIDS)*np.ones(self.NUM_BIDS)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])

        self.last_pull = None
        self.a_hist = []

    def bid(self, value, context, estimated_CTR):
        pulled_arm = self.rng.choice(self.NUM_BIDS, p=self.probabilities, size=None)
        self.last_pull = pulled_arm
        self.a_hist.append(pulled_arm)
        return self.BIDS[pulled_arm]

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        rewards = surpluses / values       # rewards are normalized, in [0,1]

        # for each bid(arm) played, update the expected reward for that bid(arm)
        for i, bid in enumerate(bids):
            arm = np.where(self.BIDS == bid)[0][0]
            self.est_rewards[arm] = rewards[i] / self.probabilities[arm]
            self.w[arm] *= np.exp(self.gamma * self.est_rewards[arm] / self.NUM_BIDS)
            self.w[~np.isfinite(self.w)] = 0
            self.probabilities = (1 - self.gamma) * self.w / self.w.sum()  +  self.gamma / self.NUM_BIDS
            self.probabilities[0] = 1 - sum(self.probabilities[1:])


################################
######      Exp3 IX       ######
################################
class Exp3IX(BaseBidder):
    def __init__(self, rng, learning_rate=1):
        super(Exp3IX, self).__init__(rng)
        '''
        learning_rate = min(1, cubic_root( (K * ln K)/(2 * g) ))
            with K being the number of arms
            with g being an upper bound of the sum of all rewards the best algorithm can get
                in my case  g = value * total_num_auctions
        '''
        self.learning_rate = 0.05   # learning_rate = cubic_root( (11 * ln11)/(2 * 118'000) )
        self.gamma = self.learning_rate / 2

        self.max_weight = 1e4

        self.L = np.zeros(self.NUM_BIDS)    # avg loss (regret) instead of reward
        self.w = np.ones(self.NUM_BIDS)
        self.p = np.ones(self.NUM_BIDS, dtype=np.float64) / self.NUM_BIDS
        self.p[0] = 1 - self.p[1:].sum()    # make sure that sum(p)=1 

    def bid(self, value, context, estimated_CTR):
        if (self.p<0).any():
            raise ValueError("Negative probability in Exp3: ", self.p)
        pulled_arm = self.rng.choice(self.NUM_BIDS, p=self.p)
        return self.BIDS[pulled_arm]
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        rewards = surpluses / values       # rewards are normalized, in [0,1]

        # for each bid(arm) played, update the expected reward for that bid(arm)
        for i, bid in enumerate(bids):
            arm_id = np.where(self.BIDS == bid)[0][0]
            # self.L[arm_id] +=  (1 - rewards[i]) / (self.p[arm_id] + self.gamma)
            # self.w[arm_id] = np.exp(-1 * self.learning_rate * self.L[arm_id])
            # self.w[~np.isfinite(self.w)] = 0    # disactivate arms with infinite weight
            # self.p[arm_id] = self.w[arm_id] / self.w.sum()

            ## Modified as Gianmarco Exp3
            self.L[arm_id] +=  (1 - rewards[i]) / (self.p[arm_id])
            self.w[arm_id] = np.exp(-1 * self.learning_rate * self.L[arm_id] / self.NUM_BIDS)
            self.w[~np.isfinite(self.w)] = 0    # disactivate arms with infinite weight
            self.p = (1 - self.learning_rate) * self.w / self.w.sum()  +  self.learning_rate / self.NUM_BIDS
            assert self.p.sum() == 1.0, f"Sum of probabilities is not 1, {self.p} sums to {self.p.sum()}"
        
        p0 = self.p[0]
        self.p[0] = 1 - self.p[1:].sum()
        if np.abs(p0-self.p[0]) < 0.01:
            raise ValueError("p0 changed too much", p0, 'vs', self.p[0], '\n\n', self.p)
        if self.p.sum() > 1.0:
            print("Sum of probabilities is not 1", self.p)
            self.p = self.p / self.p.sum()


################################
#####       GP_UCB        ######
################################

### suorce: https://github.com/tushuhei/gpucb

from sklearn.gaussian_process import GaussianProcessRegressor
class gp_ucb(BaseBidder):
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
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses, estimated_CTRs)
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

class warm_start_gpr(BaseBidder):
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
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses, estimated_CTRs)
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
class IGPRBidder(BaseBidder):
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
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses, estimated_CTRs)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!

        # partial_fit
        # self.X.append(bids)
        # self.Y.append(surpluses)
        pass
        # for x, y in zip(self.X, self.Y):
        for i in range(len(bids)):
            self.igpr.learn(new_x=bids[i], new_y=surpluses[i])


######################################
#####   batch incremental GPR   ######
######################################

from ModelsMine import BIGPR
class BIGPRBidder(BaseBidder):
    def __init__(self, rng, arms_amount=20, max_k_matrix_size=2000):
        super(BIGPRBidder, self).__init__(rng)

        self.bigpr = BIGPR( init_x=np.array([0.], dtype=np.float32), init_y=np.array([0.], dtype=np.float32),
                            max_k_matrix_size=max_k_matrix_size  )

        self.fit_once = False

    def bid(self, value, context, estimated_CTR):
        if self.fit_once:
            expected_rewards = np.zeros_like(self.BIDS)
            for i, bid in enumerate(self.BIDS):
                expected_rewards[i] = self.bigpr.predict(bid)
            chosen_bid = self.BIDS[np.argmax(expected_rewards)]
        else:
            chosen_bid = self.rng.choice(self.BIDS)
        return chosen_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]

        x = bids.reshape(-1,1).astype(np.float32)
        y = surpluses.reshape(-1,1).astype(np.float32)
        self.bigpr.learn_batch(new_xs=x, new_ys=y)
