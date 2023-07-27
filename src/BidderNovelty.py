import os
from BidderBandits import BaseBandit
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from ModelsMine import CVR_Estimator, IGPR, BIGPR
from sklearn.linear_model import SGDRegressor
import time     # for saving models
import joblib   # for saving models
from utils import get_project_root

ROOT_DIR = get_project_root()


class NoveltyBidder(BaseBandit):
    '''
    The reward (in this setting the surplus) is calculated as follows
    
    Reward = ( conv(x)*value-price(a) ) * win(x,a)
    
    - win is a binary variable expressing win
    - conv(x) conversion probability estimated by a regressor

    we observed how the function could be decoupled
    and one could learn c(x) indipendently
    and then optimize for the bid, `a`
    '''
    def __init__(self, rng, isContinuous=False, textContinuous="computes Continuous Actions"):
        super().__init__(rng, isContinuous, textContinuous)


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
        
        return conv_prob * value  # TODO: choose bid more wisely!!

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        surpluses = np.zeros_like(values)
        surpluses[won_mask] = values[won_mask] * outcomes[won_mask] - prices[won_mask]
                
        # IN HINDISGHT
        actions_rewards, regrets = self.calculate_regret_in_hindsight_discrete(bids, values, prices, surpluses)
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

        self.cvr_regressor = BIGPR(init_x=np.array([0., 0., 0., 0., 0., 1.]), init_y=np.array([0.5]))
        self.bid_regressor = BIGPR(init_x=np.array([0.0, 0.0]), init_y=np.array([0.0]))

    def bid(self, value, context, estimated_CTR):
        cvr = self.cvr_regressor.predict(context.reshape(1,-1).astype(np.float64))[0]
        bid = self.bid_regressor.predict(np.array([value, cvr], dtype=np.float64).reshape(1,-1))[0]
        return bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   dont use the whole dataset. only when [won_mask]. outcomes when won_mask=0 are based on other agents' products
        X_cvr = contexts[won_mask].astype(np.float64)
        y_cvr = outcomes[won_mask].reshape(-1,1).astype(np.float64)
        if X_cvr.size > 0:
            self.cvr_regressor.learn_batch(X_cvr, y_cvr)
        
        # update bandit bidder
        #   can use the whole dataset. values, cvrs refer to my product. target bid is given by regret in hindisght computation
        cvrs = self.cvr_regressor.predict(contexts.astype(np.float64))
        X_bid = np.array([values, cvrs]).T.astype(np.float64)
        best_bids = actions_rewards[:, 0]
        y_bid = best_bids.reshape(-1,1).astype(np.float64)
        if X_bid.size > 0:
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
    def __init__(self, rng):
        super(NoveltyBidderSGD, self).__init__(rng, isContinuous=True)

        self.random_state = rng.choice(100)
        self.cvr_regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.]], [0.5])   #ctxt (6,) -> cvr (1,) 
        self.bid_regressor = SGDRegressor(random_state=self.random_state).fit([[0.0, 0.0]], [0.0])  # value, cvr (2,) -> bid (1,)
    
    def bid(self, value, context, estimated_CTR):
        cvr = self.cvr_regressor.predict(context.reshape(1,-1))[0]
        bid = self.bid_regressor.predict(np.array([value, cvr]).reshape(1,-1))[0]
        return bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   i shouldnt use the whole dataset.
        #   only when i win the outcomes is referred to me!
        #   if a conversion is done on someone else's product it's useless
        X_cvr = contexts[won_mask]
        y_cvr = outcomes[won_mask]
        if X_cvr.size > 0:
            # X = X.reshape(-1, 1) -> X has already 2 dims
            # y = y.reshape(-1, 1)
            self.cvr_regressor.partial_fit(X_cvr, y_cvr)
            
        # update bandit bidder
        #   i can use the whole dataset! from ctxt i predict the cvr, the values are always referred to me, so i can predict bids
        cvrs = self.cvr_regressor.predict(contexts)
        X_bid = np.array([values, cvrs]).T
        best_bids = actions_rewards[:, 0]
        y_bid = best_bids
        if X_bid.size > 0:
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
        super(NoveltyBidderSGD, self).__init__(rng, isContinuous=True)

        self.random_state = rng.choice(100)
        # self.cvr_regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.]], [0.5])   #ctxt (6,) -> cvr (1,) 
        # self.bid_regressor = SGDRegressor(random_state=self.random_state).fit([[0.0, 0.0]], [0.0])  # value, cvr (2,) -> bid (1,)
        self.regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.], [0.0]], [0.0])   #ctxt (6,), value (1,) -> bid (1,)
    
    def bid(self, value, context, estimated_CTR):
        bid = self.regressor.predict(np.array([context, value]).reshape(1,-1))[0]
        return bid
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        actions_rewards, regrets = super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        #   i shouldnt use the whole dataset.
        #   only when i win the outcomes is referred to me!
        #   if a conversion is done on someone else's product it's useless
        X1 = contexts[won_mask]
        X2 = values[won_mask]
        X = np.array([X1, X2]).T
        optimal_bids = actions_rewards[won_mask, 0]
        y = optimal_bids
        assert X1.size == X2.size == y.size
        if y.size > 0:
            # X = X.reshape(-1, 1) -> X has already 2 dims
            # y = y.reshape(-1, 1)
            self.regressor.partial_fit(X, y)


        if self.save_model and iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            foldername = "src/models/sgd/"
            os.makedirs(ROOT_DIR / foldername, exist_ok=True)
            joblib.dump(self.regressor, ROOT_DIR / foldername / (ts+".joblib") )
