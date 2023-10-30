import os
from BidderBandits import BaseBidder
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from ModelsMine import CVR_Estimator, IGPR, BIGPR
from sklearn.linear_model import SGDRegressor
import time     # for saving models
import joblib   # for saving models
from utils import get_project_root

ROOT_DIR = get_project_root()


class NoveltyBidder(BaseBidder):
    '''
    The reward (in this setting the surplus) is calculated as follows
    
    Reward = ( ctr(x) * value-price(a) ) * win(x,a)
    
    - win is a binary variable expressing win
    - conv(x) conversion probability estimated by a regressor

    we observed how the function could be decoupled
    and one could learn c(x) indipendently
    and then optimize for the bid, `a`
    '''
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions"):
        super().__init__(rng, isContinuous, textContinuous)


#import lasso regression from sklearn.linear_model
from sklearn.linear_model import Lasso, Ridge
class NoveltyClairevoyant(NoveltyBidder):
    '''
    Clairevoyant Bidder base class
    '''
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions"):
        super(NoveltyClairevoyant, self).__init__(rng, isContinuous, textContinuous)


################################
###   Novelty CV mkt-price   ###
################################
class NoveltyClairevoyant_mktprice(NoveltyClairevoyant):
    def __init__(self, rng, how_many_temp=100):
        super(NoveltyClairevoyant_mktprice, self).__init__(rng, isContinuous=False)
        self.random_state = rng.choice(100)
        self.contexts = []
        # self.bids_surpluses = [[] for _ in range(self.NUM_BIDS)]
        self.mkt_prices = []
        self.best_bids = []

        # self.how_many_temp = how_many_temp
        self.ts = time.strftime("%Y%m%d-%H%M", time.localtime())

    def bid(self, value, context, estimated_CTR):
        return 0.0
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        
        # SAVE DATA

        # mkt_price is the price of the highest bid except mine
        #   so self.winning_bids or self.second_winning_bids by comparing self.winning_bids with bids
        mkt_prices = self.winning_bids
        new_contexts = contexts.copy()

        self.mkt_prices.extend(list(mkt_prices))
        self.contexts.extend(list(new_contexts))

        # super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        ###############################
        #### Closed-Form BEST BIDS ####
        ###############################
        best_bids = np.array(self.winning_bids, dtype=np.float32) + 0.01

        # discretize best_bids into self.BIDS values
        pass
        best_bids = np.array([ np.min(  np.concatenate((self.BIDS[self.BIDS-bid > 0.0], [np.inf]))  ) for bid in best_bids ])
        
        best_bids[best_bids > values] = 0.0
        # self.best_bids.extend(list(best_bids))

        # AND THAT'S ALL FOLKS!
        #  now calculate the updated regret in hindsight to be plotted
        #  and the updated actions_rewards to be averaged
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]
        #FIXME: should pass expected_surpluses to calculate_regrets, but since all bids are 0.0 should be the same
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(best_bids, values, prices, surpluses, estimated_CTRs)
        self.regret.extend(regrets)
        self.surpluses.extend(surpluses)
        self.actions_rewards.extend(actions_rewards)

        if iteration == self.num_iterations-1:
            print("Clairevoyant: training model")
            print("Contexts amount: {}".format(len(self.contexts)))
            print("Mkt_prices amount: {}".format(len(self.mkt_prices)))

            # make the model learn
            X = self.contexts
            y = self.mkt_prices

            # X.reshape(-1, 1)
            # y.reshape(-1, 1)
            print("now")
            regressor = Ridge(alpha = 1e-10 ,random_state=self.random_state).fit(X, y)
            # TODO: from mkt_price to best_bid just + 0.01
            print("over")

            #save the model for later use
            foldername = f"src/models/clairevoyant/mkt_price/{self.ts}"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)

            print("Saving model in {}".format(ROOT_DIR / foldername / (self.ts+".joblib")))
            joblib.dump(regressor, ROOT_DIR / foldername / (self.ts+".joblib") )

            contexts_data = np.array(self.contexts)
            np.save(ROOT_DIR / foldername / "contexts.npy", contexts_data)
            mkt_prices_data = np.array(self.mkt_prices)
            np.save(ROOT_DIR / foldername / "_mkt_prices.npy", mkt_prices_data)

#########################################
###  Novelty CV discrete contextual   ###
#########################################
class NoveltyClairevoyant_discr_ctxt(NoveltyClairevoyant):
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions", observable_context_dim=1):
        super().__init__(rng, isContinuous, textContinuous)

        self.contexts = []
        # self.optimal_bids = []        #useless

        self.ctrs_n = []
        self.win_bids_n = []

        # NOTE: Compute won_mask and bids -> ex post
        #       bc: 
        #           1. cv doesnt bid, so bids=0, wonmask=False
        #           2. even if it bid, i would have data for 1 arm, but i want winprob for each arm!!

        # self.won_mask_n = []  #NOTE: USELESS! bc cv doesnt bid
        # self.bids = []  # already stored by parent class

        self.ts = time.strftime("%Y%m%d-%H%M", time.localtime())

        self.c_dims = observable_context_dim

    def bid(self, value, context, estimated_CTR):
        return 0.0    
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        
        # best_bids = np.array(self.winning_bids, dtype=np.float32)
        # best_bids_discr = np.array([ np.min(  np.concatenate((self.BIDS[self.BIDS-bid > 0.0], [np.inf]))  ) for bid in best_bids ])
        # best_bids_discr[best_bids_discr > values] = 0.0
        # self.optimal_bids.extend(list(best_bids_discr))

        self.ctrs_n.extend(list(estimated_CTRs))
        self.win_bids_n.extend(list(self.winning_bids))

        #self.contexts already saved by parent class
        #self.bids already saved by parent class

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        if iteration == self.num_iterations-1:
            print(iteration, self.num_iterations)
            #TODO: MODIFY -> GET CONTEXTs -> BEST BID BUT FOR REAL
            '''
            1. create context masks
            2. for each context (uniques)
            3. for each bid
            4. compute avg_reward_bid
                (v * ctr - bid) * win_prob

            so i need:
                - value
                - estimated_CTRs
                - won_mask
                - bids

            '''

            contexts = np.array(self.contexts, dtype=np.float32)
            obs_contexts = contexts[:,0:self.c_dims]

            # contexts_set = set(obs_contexts)
            contexts_set = np.unique(obs_contexts, axis=0)
            # contexts_masks = {
            #     ctxt: obs_contexts == ctxt for ctxt in contexts_set
            # }
            contexts_masks = np.array([
                np.equal(obs_contexts, ctxt).all(axis=1) for ctxt in contexts_set
            ])

            # best_bids_overall = {
            #     ctxt: 0.0 for ctxt in contexts_set
            # }
            best_bids_overall = np.zeros(len(contexts_set))
            # avg_bids_surpluses = {
            #     ctxt: [0. for _ in self.BIDS] for ctxt in contexts_set
            # }
            avg_bids_surpluses = np.zeros((len(contexts_set), len(self.BIDS)))
            avg_bids_surpluses2 = np.zeros((len(contexts_set), len(self.BIDS)))
            avg_bids_surpluses3 = np.zeros((len(contexts_set), len(self.BIDS)))

            value = values[0]   # ASSUMPTION OF SINGLE PRODUCT
            CTRs = np.array(self.ctrs_n)  #TODO: do i need to mask the ctrs with won_mask??? can i use gym info that a bidder couldnt use?
            # SAVE ALSO WINNING BIDS, (THAT ARE ALSO PRICES IF I WIN)
            win_bids_n = np.array(self.win_bids_n)

            for i_ctxt, ctxt in enumerate(contexts_set):
                ctxt_mask = contexts_masks[i_ctxt]

                arms_utilities_in_hs = np.array(self.arms_utility_in_hindsight)

                ctxt_CTR = CTRs[ctxt_mask].mean()
                ctxt_winbids = win_bids_n[ctxt_mask]

                for i_bid, bid in enumerate(self.BIDS):
                    arm_utilities = arms_utilities_in_hs[ctxt_mask][:,i_bid]
                    avg_bids_surpluses[i_ctxt][i_bid] = np.mean(arm_utilities)

                    ctxt_bid_winprob = np.mean(bid >= ctxt_winbids)

                    avg_bids_surpluses2[i_ctxt][i_bid] = (value * ctxt_CTR - bid) * ctxt_bid_winprob
                    # avg_bids_surpluses[bid_mask][i] = np.mean(optimal_surpluses[bid_mask]) if np.sum(bid_mask) > 0 else 0.0
                
                best_bids_overall[i_ctxt] = self.BIDS[np.argmax(avg_bids_surpluses[i_ctxt])]

            print("Clairevoyant results:")
            for i_ctxt, ctxt in enumerate(contexts_set):
                print("context: {}".format(contexts_set[i_ctxt]))
                for i_bid in range(len(self.BIDS)):
                    print(f"\tbid: {self.BIDS[i_bid]:.2f} \t avg_surplus: {avg_bids_surpluses[i_ctxt][i_bid]}")
                    print(f"\tbid: {self.BIDS[i_bid]:.2f} \t avg_surplus2: {avg_bids_surpluses2[i_ctxt][i_bid]}")
            print("best_bid_overall: {}".format(best_bids_overall))


            # np.save best_bid_overall
            foldername = f"src/models/clairevoyant/best_bid_ctxt/{self.ts}"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)

            # write it in a file
            with open(ROOT_DIR / foldername / "clairevoyant_bestbid_ctxt.txt", "w") as f:
                for i_ctxt, ctxt in enumerate(contexts_set):
                    f.write("context: {}\n".format(ctxt))
                    for i_bid in range(len(self.BIDS)):
                        f.write(f"\tbid: {self.BIDS[i_bid]:.2f} \t avg_surplus: {avg_bids_surpluses[i_ctxt][i_bid]}\n")
                f.write("best_bid_overall: {}\n".format(best_bids_overall))

            # x = self.rng.choice(list(contexts_set), size=(200,))
            context_vals = np.unique(contexts[:,0])
            # x = np.array(np.meshgrid(*[[ctxt_val for ctxt_val in context_vals] for _ in range(5)], [1.0])).T.reshape(-1,6)
            # y = np.array([ best_bids_overall[np.where(np.equal(contexts_set, c).all(axis=1))] - 0.01 for c in x[:,0:self.c_dims] ]).reshape(-1,1)
            # model = Ridge().fit(x, y)
            # joblib.dump(model, ROOT_DIR / foldername / ("clairevoyant_bestbid_ctxt.joblib") )
            # print("saved model in {}".format(ROOT_DIR / foldername / ("clairevoyant_bestbid_ctxt.joblib") ))
            np.save(ROOT_DIR / foldername / "clairevoyant_ctxt_bestbid.npy", (contexts_set.squeeze(), best_bids_overall))
            print("saved model in {}".format(ROOT_DIR / foldername / ("clairevoyant_ctxt_bestbid.npy") ))


###############################
###   Novelty CV best-bid   ###
###############################
class NoveltyClairevoyant_bestbid(NoveltyClairevoyant):
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions"):
        super(NoveltyClairevoyant_bestbid, self).__init__(rng, isContinuous, textContinuous)

        self.optimal_bids = []
        self.ts = time.strftime("%Y%m%d-%H%M", time.localtime())
    
    def bid(self, value, context, estimated_CTR):
        return 0.0
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        best_bids = np.array(self.winning_bids, dtype=np.float32)
        best_bids_discr = np.array([ np.min(  np.concatenate((self.BIDS[self.BIDS-bid > 0.0], [np.inf]))  ) for bid in best_bids ])
        best_bids_discr[best_bids_discr > values] = 0.0
        self.optimal_bids.extend(list(best_bids_discr))

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        if iteration == self.num_iterations-1:
            print(iteration, self.num_iterations)
            avg_bids_surpluses = [0. for _ in self.BIDS]

            arms_utilities_in_hs = np.array(self.arms_utility_in_hindsight)

            for i in range(len(self.BIDS)):
                arm_utilities = arms_utilities_in_hs[:,i]
                avg_bids_surpluses[i] = np.mean(arm_utilities)

            best_bid_overall = self.BIDS[np.argmax(avg_bids_surpluses)]

            print("Clairevoyant results:")
            for i in range(len(self.BIDS)):
                print("bid: {} \t avg_surplus: {}".format(self.BIDS[i], avg_bids_surpluses[i]))
            print("best_bid_overall: {}".format(best_bid_overall))

            # np.save best_bid_overall
            foldername = f"src/models/clairevoyant/best_bid/{self.ts}"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)

            # x = ( self.rng.uniform(-10, 10, (50,6)) ) 
            # y = ( np.zeros(x.shape[0]) + best_bid_overall - 0.01 )
            # model = Ridge().fit(x, y)

            # joblib.dump(model, ROOT_DIR / foldername / ("clairevoyant_bestbid.joblib") )
            np.save(ROOT_DIR / foldername / "clairevoyant_bestbid.npy", best_bid_overall)

###
### CTR regression GPR
###
class NoveltyBidderGPR(NoveltyBidder):
    def __init__(self, rng, regression_model='IGPR'):
        super(NoveltyBidderGPR, self).__init__(rng, isContinuous=True)
        '''
        self.estimator = CVR_Estimator(eval(regression_model))
        self.conversion_estimator = IGPR(init_x=np.array([0., 0., 0., 0., 0., 1.]), init_y=0.5)
        '''

        self.cvr_regressor_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        self.random_state = rng.choice(100)
        self.cvr_regressor = GaussianProcessRegressor(kernel=self.cvr_regressor_kernel, random_state=self.random_state)\
                                .fit([[0., 0., 0., 0., 0., 1.]], [0.5])


    def bid(self, value, context, estimated_CTR):
        conv_prob = self.cvr_regressor.predict(context.reshape(1, -1))[0]
        # with discrete arms -> c*value - arm
        # with continuous arms -> Regression: (value, conv_prob) => bid
        
        return conv_prob * value    #TRUTHFUL BIDDING, but it shouldnt
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]
                
        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses, estimated_CTRs)
        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!

        # update regression model
        # i shouldnt use the whole dataset, but only when i win
        X = contexts[won_mask]
        y = outcomes[won_mask]
        if X.size > 0:
            if len(X.shape) == 1:
                X.reshape(1, -1)
                y.reshape(1, -1)
            else:
                X.reshape(-1, 1)
                y.reshape(-1, 1)
            '''
            for x, y in zip(my_contexts, my_outcomes):
                self.cvr_regressor.learn(x, y) # but i want conv prob!!!
            '''
            gpr = GaussianProcessRegressor(kernel=self.cvr_regressor_kernel, random_state=self.random_state).fit(X, y)

            self.cvr_regressor = gpr
            self.cvr_regressor_kernel.set_params(**(gpr.kernel_.get_params()))

        if iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/gpr/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.cvr_regressor, ROOT_DIR / foldername / (ts+".joblib") )


        # update bandit bidder (exp3???????)
        pass # for now truthful bidding


###
### CTR regression BIGPR
###
class NoveltyBidderBIGPR(NoveltyBidder):
    '''
    '''
    def __init__(self, rng, isContinuous=True):
        super().__init__(rng, isContinuous)

        # self.cvr_regressor = BIGPR(init_x=np.array([0., 0., 0., 0., 0., 1.]), init_y=np.array([0.5]))
        self.cvr_regressor = None
        # self.bid_regressor = BIGPR(init_x=np.array([0.5, 1.0]), init_y=np.array([1.0]))
        self.bid_regressor = None

    def bid(self, value, context, estimated_CTR):
        if self.cvr_regressor is None:
            return self.rng.uniform(0, value)
        
        cvr = self.cvr_regressor.predict(context.reshape(1,-1).astype(np.float32))[0]
        bid = self.bid_regressor.predict(np.array([value, cvr], dtype=np.float32).reshape(1,-1))[0]
        return bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   dont use the whole dataset. only when [won_mask]. outcomes when won_mask=0 are based on other agents' products
        X_cvr = contexts[won_mask].astype(np.float32)
        y_cvr = outcomes[won_mask].reshape(-1,1).astype(np.float32)
        if X_cvr.size > 0:
            if self.cvr_regressor is None:
                self.cvr_regressor = BIGPR(init_x=X_cvr[0], init_y=y_cvr[0])
                self.cvr_regressor.learn_batch(X_cvr[1:], y_cvr[1:])
            else:
                self.cvr_regressor.learn_batch(X_cvr, y_cvr)
        
        # update bandit bidder
        #   can use the whole dataset. values, cvrs refer to my product. target bid is given by regret in hindisght computation
        cvrs = self.cvr_regressor.predict(contexts.astype(np.float32))
        X_bid = np.array([values, cvrs]).T.astype(np.float32)
        best_bids = actions_rewards[:, 0]
        y_bid = best_bids.reshape(-1,1).astype(np.float32)
        if X_bid.size > 0:
            if self.bid_regressor is None:
                self.bid_regressor = BIGPR(init_x=X_bid[0], init_y=y_bid[0])
                self.bid_regressor.learn_batch(X_bid[1:], y_bid[1:])
            else:
                self.bid_regressor.learn_batch(X_bid, y_bid)
        
        if iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/bigpr/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.cvr_regressor, ROOT_DIR / foldername / (ts+".joblib") )


###
### CTR regression SGD
###
class NoveltyBidderSGD(NoveltyBidder):
    '''
    #  context (6,)  --->  [cvr_regressor] ---> cvr (1,)  \   
    #                                                      \
    #                                                       |---> [bid_regressor] --->  bid (1,)
    #                                                      /
    #                                         value (1,)  /
    '''
    def __init__(self, rng, nsteps=12):
        super(NoveltyBidderSGD, self).__init__(rng, isContinuous=False)

        # self.BIDS = np.linspace(0.01, 1., nsteps)     #used as percentage of maxbid (being 1.5 * value)

        self.random_state = rng.choice(100)
        # self.cvr_regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.]], [0.5])   #ctxt (6,) -> cvr (1,) 
        self.cvr_regressor = None
        # self.bid_regressor = SGDRegressor(random_state=self.random_state).fit([[0.5, 1.0]], [3.0])  # value, cvr (2,) -> bid (1,)
        self.bid_regressor = None

        # CLAIREVOYANT moved in parent class
        self.clairevoyant = joblib.load(ROOT_DIR / "src/models/clairevoyant/20230905-1417.joblib")
    
    def bid(self, value, context, estimated_CTR):
        if self.cvr_regressor is None:
            return self.rng.choice(self.BIDS) * value * 1.5
        cvr = self.cvr_regressor.predict(context.reshape(1,-1))[0]
        bid = self.bid_regressor.predict(np.array([value, cvr]).reshape(1,-1))[0]
        closest_bid = self.BIDS[np.argmin(np.abs(self.BIDS - bid))]
        return closest_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   i shouldnt use the whole dataset.
        #   only when i win the outcomes is referred to me!
        #   if a conversion is done on someone else's product it's useless
        X_cvr = contexts[won_mask]
        y_cvr = outcomes[won_mask]

        if X_cvr.size > 0:
            if self.cvr_regressor is None:
                self.cvr_regressor = SGDRegressor(random_state=self.random_state).fit(X_cvr, y_cvr)
            else:
                # X = X.reshape(-1, 1) -> X has already 2 dims
                # y = y.reshape(-1, 1)
                self.cvr_regressor.partial_fit(X_cvr, y_cvr)
            
        # update bandit bidder
        #   i can use the whole dataset! from ctxt i predict the cvr, the values are always referred to me, so i can predict bids
        cvrs = self.cvr_regressor.predict(contexts)
        X_bid = np.array([values, cvrs]).T


        # best_bids = actions_rewards[:, 0]
        # y_bid = best_bids
        # NEW BEST BIDS, CALL CLAIREVOYANT
        mkt_prices = self.clairevoyant.predict(contexts)

        surpluses_hs = np.array(  [ [ (bid>mkt_prices[i])*(values[i]-bid)*estimated_CTRs[i]  for  bid in self.BIDS] for i in range(values.shape[0]) ]  )
        max_surpluses_hs = np.max(surpluses_hs, axis=1)
        best_bids = self.BIDS[np.argmax(surpluses_hs, axis=1)]
        best_bids[max_surpluses_hs <= 0] = 0.0

        y_bid = best_bids
        
        if X_bid.size > 0:
            if self.bid_regressor is None:
                self.bid_regressor = SGDRegressor(random_state=self.random_state).fit(X_bid, y_bid)
            else:
                self.bid_regressor.partial_fit(X_bid, y_bid)

        if iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/sgd/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.cvr_regressor, ROOT_DIR / foldername / (ts+".joblib") )


###
### CTR regression Neural Network
###
import torch as th
class NoveltyBidderNN(NoveltyBidder):
    def __init__(self, rng, epochs=256, device=None, pretrained_model=None):
        super(NoveltyBidderNN, self).__init__(rng, isContinuous=True)

        self.random_state = rng.choice(100)
        if pretrained_model is not None:
            self.cvr_regressor = th.load(pretrained_model)
        else:
            self.cvr_regressor = th.nn.Sequential(
                                                th.nn.Linear(6,4),
                                                th.nn.Dropout(0.4),
                                                th.nn.ReLU(),
                                                th.nn.Linear(4,2),
                                                th.nn.Dropout(0.4),
                                                th.nn.ReLU(),
                                                th.nn.Linear(2,1),
                                                th.nn.ReLU(),
                                                th.nn.Sigmoid()      )
            
        self.device = device if device is not None else th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.loss = th.nn.MSELoss()
        self.optimizer = th.optim.Adam(self.cvr_regressor.parameters(), lr=0.001)
        self.epochs = epochs

        self.contexts = []
        self.outcomes = []

    def bid(self, value, context, estimated_CTR):
        conv = self.cvr_regressor(th.tensor(context, dtype=th.float32)).item()
        return conv * value
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # save data
        self.contexts.extend(contexts[won_mask])
        self.outcomes.extend(outcomes[won_mask])

        # update nn
        # i shouldnt use the whole dataset, but only when i win

        if len(self.contexts) > 0:
            X = np.array(self.contexts, dtype=np.float32)
            y = np.array(self.outcomes, dtype=np.float32).reshape(-1,1)        
            X = th.tensor(X, device=self.device, dtype=th.float32)
            th.nn.functional.normalize(X, p=2, dim=1, eps=1e-12, out=X)
            y = th.tensor(y, device=self.device, dtype=th.float32)

            self.cvr_regressor.to(self.device)
            self.cvr_regressor.train()
            
            for i in range(self.epochs):
                self.optimizer.zero_grad()
                y_pred = self.cvr_regressor(X)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optimizer.step()

            if iteration == self.num_iterations-1:
                ts = time.strftime("%Y%m%d-%H%M", time.localtime())
                foldername = "src/models/nn_6-4-2-1/"
                os.makedirs(ROOT_DIR / foldername, exist_ok=True)
                th.save(self.cvr_regressor, ROOT_DIR / foldername / (ts+".pt") )


###
### Direct Prediction SGD
###

class NoveltyDirectSGD(NoveltyBidder):
    '''
    #   context (6,)  \   
    #                  \
    #                   |---> [bid_regressor] --->  bid  (1,)
    #                  /
    #    value (1,)   /
    '''
    def __init__(self, rng):
        super(NoveltyDirectSGD, self).__init__(rng, isContinuous=True)

        self.save_model = False
        self.random_state = rng.choice(100)
        # self.cvr_regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.]], [0.5])   #ctxt (6,) -> cvr (1,) 
        # self.bid_regressor = SGDRegressor(random_state=self.random_state).fit([[0.0, 0.0]], [0.0])  # value, cvr (2,) -> bid (1,)
        # self.regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1., 1.]], [1.0])   #ctxt (6,), value (1,) -> bid (1,)
        self.regressor = None
    
    def bid(self, value, context, estimated_CTR):
        if self.regressor is None:
            return self.rng.uniform(0, value)
        
        x = np.concatenate([context, [value]]).reshape(1,-1)
        bid = self.regressor.predict( x )[0]
        return bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   i shouldnt use the whole dataset.
        #   only when i win the outcomes is referred to me!
        #   if a conversion is done on someone else's product it's useless
        X1 = contexts[won_mask]
        X2 = values[won_mask].reshape(-1,1)
        optimal_bids = actions_rewards[won_mask, 0]
        y = optimal_bids
        assert X1.shape[0] == X2.shape[0] == y.size, "X1.shape[0]={}, X2.shape[0]={}, y.size={}".format(X1.shape[0], X2.shape[0], y.size)
        if y.size > 0:
            X = np.concatenate([X1, X2], axis=1)
            if self.regressor is None:
                self.regressor = SGDRegressor(random_state=self.random_state).fit(X, y)
            else:
                # X = X.reshape(-1, 1) -> X has already 2 dims
                # y = y.reshape(-1, 1)
                self.regressor.partial_fit(X, y)


        if self.save_model and iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/sgd/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.regressor, ROOT_DIR / foldername / (ts+".joblib") )


###
### Direct SGD with use of estimatedCTR
###

class NoveltyDirectSGD_wCTR(NoveltyDirectSGD):
    def calculate_regret_in_hindsight_continuous(self, bids, values, prices, outcomes, estimatedCTRs, surpluses):
        
        actions_rewards = np.zeros((values.size, 2))      # tuples (arm, reward)

        for i in range(len(values)):
            # bid_to_beat is the bid that would have won the auction, excluding mine since i am recalculating it
            bid_to_beat = self.winning_bids[i]  if self.winning_bids[i] != bids[i]  else self.second_winning_bids[i]

            win_bid = bid_to_beat + 0.01
            price_if_win = win_bid  if self.auction_type == 'FirstPrice'  else bid_to_beat # SecondPrice
            reward = max( 0 ,  values[i] - price_if_win )
            best_bid = win_bid*estimatedCTRs[i]  if reward > 0  else 0.0  #should never happen that price > value
            actions_rewards[i] = (best_bid, reward)

        regrets = actions_rewards[:, 1] - surpluses
        return actions_rewards, regrets
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]

        actions_rewards, regrets = self.calculate_regret_in_hindsight_continuous(bids, values, prices, outcomes, estimated_CTRs, surpluses)

        self.regret.append(regrets.sum())
        self.actions_rewards.append(actions_rewards)    # batch not averaged !!!

        # update regression model
        #   i shouldnt use the whole dataset.
        #   only when i win the outcomes is referred to me!
        #   if a conversion is done on someone else's product it's useless
        X1 = contexts[won_mask]
        X2 = values[won_mask].reshape(-1,1)
        optimal_bids = actions_rewards[won_mask, 0]
        y = optimal_bids
        assert X1.shape[0] == X2.shape[0] == y.size, "X1.shape[0]={}, X2.shape[0]={}, y.size={}".format(X1.shape[0], X2.shape[0], y.size)
        if y.size > 0:
            X = np.concatenate([X1, X2], axis=1)
            if self.regressor is None:
                self.regressor = SGDRegressor(random_state=self.random_state).fit(X, y)
            else:
                # X = X.reshape(-1, 1) -> X has already 2 dims
                # y = y.reshape(-1, 1)
                self.regressor.partial_fit(X, y)

        
        if self.save_model and iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/sgd/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.regressor, ROOT_DIR / foldername / (ts+".joblib") )



###
### Novelty Direct BIGPR
###

class NoveltyDirectBIGPR(NoveltyBidder):
    '''
    #   context (6,)  \   
    #                  \
    #                   |---> [bid_regressor] --->  bid  (1,)
    #                  /
    #    value (1,)   /
    '''
    def __init__(self, rng, max_k_matrix_size=1000):
        super(NoveltyDirectBIGPR, self).__init__(rng, isContinuous=True)

        self.max_k_matrix_size = max_k_matrix_size
        self.save_model = False
        self.random_state = rng.choice(100)
        # self.cvr_regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.]], [0.5])   #ctxt (6,) -> cvr (1,) 
        # self.bid_regressor = SGDRegressor(random_state=self.random_state).fit([[0.0, 0.0]], [0.0])  # value, cvr (2,) -> bid (1,)
        # self.regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1., 1.]], [1.0])   #ctxt (6,), value (1,) -> bid (1,)
        self.regressor = None
    
    def bid(self, value, context, estimated_CTR):
        if self.regressor is None:
            return self.rng.uniform(0, value)
        
        x = np.concatenate([context, [value]]).reshape(1,-1)
        bid = self.regressor.predict( x )[0]
        return np.max( (bid, 0.0) )
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   using the whole dataset since i could always learn a bid
        X1 = contexts[won_mask]
        X2 = values[won_mask].reshape(-1,1)
        optimal_bids = actions_rewards[won_mask, 0]
        y = optimal_bids
        assert X1.shape[0] == X2.shape[0] == y.size, "X1.shape[0]={}, X2.shape[0]={}, y.size={}".format(X1.shape[0], X2.shape[0], y.size)
        if y.size > 0:
            X = np.concatenate([X1, X2], axis=1)
            if self.regressor is None:
                self.regressor = BIGPR(init_x=X[0], init_y=y[0], max_k_matrix_size=self.max_k_matrix_size)
                self.regressor.learn_batch(X[1:], y[1:])
            else:
                # X = X.reshape(-1, 1) -> X has already 2 dims
                # y = y.reshape(-1, 1)
                self.regressor.learn_batch(X, y)


        if self.save_model and iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/sgd/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.regressor, ROOT_DIR / foldername / (ts+".joblib") )


###
### UCB1 + optimism in the face of uncertainty
###

class UCB1_Optimism(NoveltyBidderSGD):
    def __init__(self, rng, regression_model="SGD"):
        super().__init__(rng, isContinuous=False)

        #TODO:
        #   - ctr regressor
        #   - bid regressor
        #   - UCB function: a in argmax(a in A) { ( ( ctr'(x) + INCctr(x))*value - price(a) ) * ( wp'(x,a) + INCwp(x,a) ) }
        # where both INC are the hoeffding bound δ * sqrt( 2 * log(t) / N(x,t) ) 

        self.regression_model = regression_model
        self.ucbs = np.zeros_like(self.BIDS)

    def bid(self, value, context, estimated_CTR):
        return super().bid(value, context, estimated_CTR)
    

###############################################
####  Novelty Uncoupled Estimation ctr, w  ####
###############################################
from BidderBandits import UCB1, Exp3
class NoveltyBidder_ctr_w_REGRESSOR(NoveltyBidder):
    '''
    Novelty model estimating CTR and win_prob indipendently

    a_t = argmax {a in A}  [ v * (ctr^ + sqrt(log(t)/N_win)) - a ] * ( w^(a) + sqrt(log(t)/N_a) ) 

    optimism bound both on ctr^ and w^

    update():
        se vinco:
            aggiorno ctr^ e N_win++
        sempre:
            aggiorno w^ e N_a++     ->     solo dell'arm giocato

    '''
    def __init__(self, rng, sub_bidder=UCB1, regression_model=SGDRegressor, observable_context_dim=1):
        super(NoveltyBidder_ctr_w_REGRESSOR, self).__init__(rng, isContinuous=True)
        self.random_state = rng.choice(100)
        self.sub_bidder_model = sub_bidder

        self.context_set = set()
        self.context_bidder = {}

        self.ctr = regression_model(random_state=self.random_state)
        self.N_ctr = 0
        self.ctr_fitted = False

        self.w_a = [regression_model(random_state=self.random_state) for _ in range(len(self.BIDS))]
        self.N_a = np.zeros(len(self.BIDS), dtype=np.int32)
        self.w_a_fitted = np.zeros(len(self.BIDS), dtype=bool)  # [False, False, ..., False]
        
        self.t  = 0
        self.c_dims = observable_context_dim

        # self.expected_rewards = np.zeros(len(self.BIDS))

    def bid(self, value, context, estimated_CTR):
        # assure every bid is played at least once before calling ctr^ w_a^
        if np.equal(self.N_a, 0).any():
            i_arm = self.rng.choice(np.where(np.equal(self.N_a, 0))[0])
            self.N_a[i_arm] += 1
            return self.BIDS[i_arm]

        # if ctr^ fitted --> then w_a are all fitted, because i bid each arm at least once (above code) 
        # if ctr^ NOT fitted --> bid trying to win, to acquire ctr data
        ctxt = context[0:self.c_dims]
        if not self.ctr_fitted:
            i_arm = self.rng.choice( range(len(self.BIDS)),  p=self.BIDS/np.sum(self.BIDS) )    # higher bid -> higher prob
            self.N_a[i_arm] += 1
            return self.BIDS[i_arm]
        
        # compute ctr^ and w_a^
        ctr = self.ctr.predict(ctxt.reshape(1,-1))[0]
        w_a = np.zeros(len(self.BIDS))
        for i, bid in enumerate(self.BIDS):
            w_a[i] = self.w_a[i].predict(ctxt.reshape(1,-1))[0]

        # expected rewards computed at every bid so that if i have more than one values, it works
        expected_rewards = np.zeros(len(self.BIDS))
        for i, bid in enumerate(self.BIDS):
            expected_rewards[i] = value * (ctr + np.sqrt(np.log(self.t)/self.N_ctr)) - bid
            expected_rewards[i] *= w_a[i] + np.sqrt(np.log(self.t)/self.N_a[i])

        # randomize if more than one optimal bid 
        best_bids_mask = expected_rewards == np.max(expected_rewards)
        best_bids = self.BIDS[best_bids_mask]
        played_bid = self.rng.choice(best_bids)
        i_arm = np.where(self.BIDS == played_bid)[0][0]
        self.N_a[i_arm] += 1
        return played_bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        '''
        se vinco:
            aggiorno ctr^ e N_win++
        sempre:
            aggiorno w^ e N_a++     ->     solo dell'arm giocato
        '''
        
        # update iteration number
        self.t += len(values)

        # update ctr^
        obs_contexts = contexts[:, 0:self.c_dims]
        n_wins = won_mask.sum()
        if n_wins > 0:
            self.N_ctr += n_wins
            x = obs_contexts[won_mask]
            # x = x.reshape(-1, 1)   if n_wins > 1     else   x.reshape(1, -1)      # with contexts[:,0] was needed, with contexts[:,0:1] no
            y = outcomes[won_mask].astype(np.float32)
            self.ctr.partial_fit(x,y)
            self.ctr_fitted = True

        # update w_a^
        for i, bid in enumerate(self.BIDS):
            mask = bids == bid
            n_plays = mask.sum()
            if n_plays > 0:
                # self.N_a[i] += n_plays    # performed in the bid() method
                x = obs_contexts[mask]
                # x = x.reshape(-1, 1)   if n_plays > 1     else   x.reshape(1, -1)     # with contexts[:,0] was needed, with contexts[:,0:1] no
                y = won_mask[mask].astype(np.float32)
                self.w_a[i].partial_fit(x, y)
                self.w_a_fitted[i] = True

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
        

        
########################
### Novelty UCB1 w  ###
########################
class NoveltyBidder_ctr_w(NoveltyBidder):
    '''
    Novelty model estimating CTR and win_prob indipendently

    a_t = argmax {a in A}  [ v * (ctr^ + sqrt(log(t)/N_win)) - a ] * ( w^(a) + sqrt(log(t)/N_a) ) 

    optimism bound both on ctr^ and w^

    update():
        se vinco:
            aggiorno ctr^ e N_win++
        sempre:
            aggiorno w^ e N_a++     ->     solo dell'arm giocato

    '''
    def __init__(self, rng, isContinuous=False, observable_context_dim=1):
        super(NoveltyBidder_ctr_w, self).__init__(rng, isContinuous)
        
        self.contexts_set = []
        self.t = 0

        # ctr
        self.N_buy = []
        self.N_win = []

        # arms
        self.N_win_a = []
        self.N_play_a = []

        self.ucbs = []

        self.c_dims = observable_context_dim

    def bid(self, value, context, estimated_CTR):
        obs_c = context[0:self.c_dims]

        if self.contexts_set==[] or not np.equal(obs_c, self.contexts_set).all(axis=1).any():
            self.contexts_set.append(obs_c)
            self.N_buy.append(0)
            self.N_win.append(0)
            self.N_win_a.append(np.zeros_like(self.BIDS, dtype=np.int32))
            self.N_play_a.append(np.zeros_like(self.BIDS, dtype=np.int32))
            self.ucbs.append(np.zeros_like(self.BIDS, dtype=np.float32))
        
        i_ctxt = np.where(np.equal(obs_c, self.contexts_set).all(axis=1))[0][0]
        if np.equal(self.N_play_a[i_ctxt], 0).any():
            i_arm = self.rng.choice(np.where(np.equal(self.N_play_a[i_ctxt], 0))[0])
            # self.N_play_a[i_ctxt][i_arm] += 1
            return self.BIDS[i_arm]

        if self.N_win[i_ctxt] == 0:
            i_arm = self.rng.choice(range(len(self.BIDS)))
            # self.N_play_a[i_ctxt][i_arm] += 1
            return self.BIDS[i_arm]
        
        # look at ucbs for i_ctxt, play max
        i_arm = np.argmax(self.ucbs[i_ctxt])
        # i_arms = np.where(np.equal(self.ucbs[i_ctxt], np.max(self.ucbs[i_ctxt])))
        # i_arm = self.rng.choice(i_arms)
        # self.N_play_a[i_ctxt][i_arm] += 1
        return self.BIDS[i_arm]
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        self.t += len(values)
        v = values[0]   # always same product

        obs_contexts = contexts[:, 0:self.c_dims]
        for i_ctxt, ctxt in enumerate(self.contexts_set):
            ctxt_mask = np.equal(obs_contexts, ctxt).all(axis=1)

            # ctr
            self.N_win[i_ctxt] += won_mask[ctxt_mask].sum()
            self.N_buy[i_ctxt] += outcomes[ won_mask & ctxt_mask ].sum()

            ctr = self.N_buy[i_ctxt] / self.N_win[i_ctxt]
            inc_ctr = np.sqrt( np.log(self.t) / self.N_win[i_ctxt] )
            
            # arms
            for i_bid, bid in enumerate(self.BIDS):
                bid_mask = np.equal(bids, bid)
                self.N_play_a[i_ctxt][i_bid] += bid_mask[ctxt_mask].sum()
                self.N_win_a[i_ctxt][i_bid] += won_mask[ bid_mask & ctxt_mask ].sum()

                win_a = self.N_win_a[i_ctxt][i_bid] / self.N_play_a[i_ctxt][i_bid]
                inc_win_a = np.sqrt( np.log(self.t) / self.N_play_a[i_ctxt][i_bid] )

                self.ucbs[i_ctxt][i_bid] = (v * (ctr+inc_ctr) - bid) * (win_a+inc_win_a)

        if iteration == self.num_iterations-1:
            print(self.contexts_set)
            for i_ctxt, ctxt in enumerate(self.contexts_set):
                print(f"{ctxt} --> {self.ucbs[i_ctxt]}")

        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)
    