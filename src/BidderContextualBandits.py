import numpy as np
import pandas as pd
from Bidder import Bidder
from BidderBandits import BaseBidder


################################
###    contextual GP_UCB     ###
################################
'''
RE-FITs GP AT EVERY ITERATION -> NOT FEASIBLE
also not contextual, estimates ctr & value ands bids their product
'''
### suorce: github.com/tushuhei/gpucb
from sklearn.gaussian_process import GaussianProcessRegressor
class gp_ucb_ctxt(BaseBidder):
    def __init__(self, rng, beta=100):
        super(gp_ucb_ctxt, self).__init__(rng)
        # self.BIDS = np.array([])
        self.beta = beta        # exploration hyperparam
        self.mu = np.array([0. for _ in range(self.NUM_BIDS)])          # ucb mean 
        self.sigma = np.array([0.5 for _ in range(self.NUM_BIDS)])      # ucb stddev
        self.X = []     # bids history
        self.Y = []     # surpluses history
        self.model = None

    def bid(self, value, context, estimated_CTR):
        if self.model is None:
            return self.rng.random()*3.0
        # idx = np.argmax(self.mu + self.sigma * np.sqrt(self.beta))
        # chosen_bid = self.BIDS[idx]
        
        return self.model.predict([context])[0]

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
        self.X.extend( contexts )
        self.Y.extend( values*estimated_CTRs )
        gp = GaussianProcessRegressor()
        # x = np.array(self.X)
        # y = np.array(self.Y).reshape(-1,1)
        gp.fit(self.X,self.Y)
        
        self.model = gp
        # data = np.array(self.BIDS).reshape(1, -1)
        # self.mu, self.sigma = gp.predict(contexts, return_std=True)    # why self.BIDS??? contexts are inputs


################################
###      expert advice       ###
################################

'''
using UCB1 as the expert
'''

from sklearn.cluster import KMeans
from BidderBandits import Exp3, UCB1
from threading import active_count, Thread
import multiprocessing as mp

class cluster_expert(BaseBidder):
    def __init__(self, rng, n_clusters=4, samples_before_clustering=1000, sub_bidder="UCB1"):
        super().__init__(rng)

        #self.BIDS = np.array([0.005, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.9, 2.4])
        #self.BIDS uguali per tutti, predefiniti 

        self.n_clusters = n_clusters
        self.samples_before_clustering = samples_before_clustering
        self.agents = [UCB1(self.rng) for _ in range(self.n_clusters)]
        self.predictor = None
        self.bid_count = 0
        self.contexts_history = []

        self.values_history = []
        self.bids_history = []
        self.prices_history = []
        self.outcomes_history = []
        self.won_mask_history = []

        self.winning_bids_history = []
        self.second_winning_bids_history = []
    
    def bid(self, value, context, estimated_CTR):
        if self.predictor is None or self.bid_count < self.samples_before_clustering:
            self.bid_count += 1
            chosen_bid = self.rng.choice(self.BIDS)
        else:
            cluster = self.predictor.predict(context.astype(np.float32).reshape(1,-1))[0]
            agent = self.agents[cluster]
            chosen_bid = agent.bid(value, context, estimated_CTR)

        self.bid_count += 1 #still counting but useless
        return chosen_bid
    
    def _update_single_agent(self, agent, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        agent.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        return

    def _update_agents(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        '''
        update each agent with its own data
        1. predict the cluster for each context
        2. for each cluster, update the corresponding agent
        '''
        agent_contexts = self.predictor.predict(contexts)
        threads = []
        # before = active_count()
        mask_n = [agent_contexts == i for i in range(self.n_clusters)]
        args_n = [(None, values[mask], bids[mask], prices[mask], outcomes[mask], None,
                    won_mask[mask], None, None, None, None, None,)      for mask in mask_n]
            
        for i, agent in enumerate(self.agents):
            agent.winning_bids = self.winning_bids[mask_n[i]]
            agent.second_winning_bids = self.second_winning_bids[mask_n[i]]
            if mask_n[i].sum() > 0:
                t = Thread(target=agent.update, args=args_n[i])
                threads.append(t)
                t.start()

        # parallellize with ray
        # for i, agent in self.agents:
        #     agent.winning_bids = self.winning_bids[mask_n[i]]
        #     agent.second_winning_bids = self.second_winning_bids[mask_n[i]]
        #     if mask_n[i].any():
        #         self._update_single_agent.remote(agent, *args_n[i])
        # processes = [self._update_single_agent.remote(agent, *args_n[i]) if mask_n[i].any() else None for i, agent in enumerate(self.agents)]
        
        # after = active_count()
        # assert after == before + len(threads), f"active threads -> before: {before}, after: {after}"
        # gives me errors idk why
        for t in threads:
            t.join()
        return

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        #save contexts
        if self.predictor is None:
            self.contexts_history.extend(contexts)

            #saving values,bids,prices,outcomes,won_mask to not waste data
            self.values_history.extend(values)
            self.bids_history.extend(bids)
            self.prices_history.extend(prices)
            self.outcomes_history.extend(outcomes)
            self.won_mask_history.extend(won_mask)
            self.winning_bids_history.extend(self.winning_bids)
            self.second_winning_bids_history.extend(self.second_winning_bids)
        
        #surplus in general
        # surpluses = np.zeros_like(values)
        # surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]

        # #regret in general
        # # IN HINDISGHT
        # actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses, estimated_CTRs)
        # self.regret.extend(regrets)
        # self.actions_rewards.extend(actions_rewards)    # not batched!!!

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        

        if self.predictor is None and self.bid_count > self.samples_before_clustering:
            # train clusters
            contexts_history = np.array(self.contexts_history, dtype=np.float32)
            self.predictor = KMeans(n_clusters=self.n_clusters).fit(contexts_history)
            # train bandits with old data
            params = []
            params.append(contexts_history)
            params.append(np.array(self.values_history, dtype=np.float32))
            params.append(np.array(self.bids_history, dtype=np.float32))
            params.append(np.array(self.prices_history, dtype=np.float32))
            params.append(np.array(self.outcomes_history))
            params.append(None)
            params.append(np.array(self.won_mask_history))
            params.append(None)
            params.append(None)
            params.append(None)
            params.append(None)
            params.append(None)

            self.winning_bids = np.array(self.winning_bids_history, dtype=np.float32)
            self.second_winning_bids = np.array(self.second_winning_bids_history, dtype=np.float32)

            self._update_agents(*params)

            # MAYBE empty history lists
            del self.contexts_history[:]
            del self.values_history[:]
            del self.bids_history[:]
            del self.prices_history[:]
            del self.outcomes_history[:]
            del self.won_mask_history[:]
            del self.winning_bids_history[:]
            del self.second_winning_bids_history[:]
            return

        if self.predictor is not None:
            self._update_agents(contexts, values, bids, prices, outcomes, None, won_mask, None, None, None, None, None)

        '''
        if predictor is not None BUT bid_count < samples_before_clustering
        then just continue to collect data
        i.e. do nothing
        '''
        return    



################################
###      PSEUDO EXPERT       ###
################################
# from BidderBandits import UCB1, Exp3, BIGPR
class PseudoExpertBidder(BaseBidder):
    def __init__(self, rng, isContinuous=False, sub_bidder=UCB1, observable_context_dim=1):
        super(PseudoExpertBidder, self).__init__(rng, isContinuous)
        self.rng = rng
        # self.n_contexts = n_contexts    #initial number of bidders
        self.sub_bidder_type = sub_bidder
        self.sub_bidders = []
        self.counters = []
        self.contexts_set = []

        # self.contexts_bidder = []

        self.c_dims = observable_context_dim

    def bid(self, value, context, estimated_CTR):
        old_context = context
        context = context[0:self.c_dims]
        if context not in self.contexts_set:
            self.contexts_set.append(context)
            # new_sub_bidder = eval(f"{self.sub_bidder_type}({self.rng}, {1.0})") #learning_rate or sigma hyperparameter
            new_sub_bidder = self.sub_bidder_type(self.rng)
            new_sub_bidder.total_num_auctions = self.total_num_auctions
            new_sub_bidder.num_iterations = self.num_iterations
            self.sub_bidders.append(new_sub_bidder)
            self.counters.append(0)
        
        i_context = self.contexts_set.index(context)
        self.counters[i_context] += 1
        return self.sub_bidders[i_context].bid(value, old_context, estimated_CTR)
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        for i_ctxt, ctxt in enumerate(self.contexts_set):
            ctxt_mask = (contexts[:,0:self.c_dims] == ctxt).squeeze()
            self.sub_bidders[i_ctxt].winning_bids = self.winning_bids[ctxt_mask]
            self.sub_bidders[i_ctxt].second_winning_bids = self.second_winning_bids[ctxt_mask]
            self.sub_bidders[i_ctxt].update(  contexts[ctxt_mask], values[ctxt_mask], bids[ctxt_mask], prices[ctxt_mask], outcomes[ctxt_mask],
                                                estimated_CTRs[ctxt_mask], won_mask[ctxt_mask], iteration, plot, figsize, fontsize, name)
            
        if iteration == self.num_iterations - 1:
            print(self.sub_bidder_type)
            print(self.counters)
            for i_ctxt, ctxt in enumerate(self.contexts_set):
                ctxt_bids = np.array(self.sub_bidders[i_ctxt].bids)
                print(f"{i_ctxt}. value:{ctxt} shape:{ctxt_bids.shape}")
                ctxt_bids_df = pd.DataFrame(ctxt_bids, columns=["bid"])
                print(ctxt_bids_df.value_counts(), '\n\n', end="")
                
