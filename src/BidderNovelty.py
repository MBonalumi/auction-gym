from BidderBandits import BaseBandit
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from ModelsMine import CVR_Estimator, IGPR
from sklearn.linear_model import SGDRegressor
import time     # for saving models
import joblib   # for saving models


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

        self.conv_regressor_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        self.random_state = rng.choice(100)
        self.conv_regressor = GaussianProcessRegressor(kernel=self.conv_regressor_kernel, random_state=self.random_state)\
                                .fit([[0., 0., 0., 0., 0., 1.]], [0.5])


    def bid(self, value, context, estimated_CTR):
        conv_prob = self.conv_regressor.predict(context.reshape(1, -1))[0]
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
                self.conv_regressor.learn(x, y) # but i want conv prob!!!
            '''
            gpr = GaussianProcessRegressor(kernel=self.conv_regressor_kernel, random_state=self.random_state).fit(X, y)

            self.conv_regressor = gpr
            self.conv_regressor_kernel.set_params(**(gpr.kernel_.get_params()))

        if iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            joblib.dump(self.conv_regressor, "./models/gpr_warmstart"+ts+".joblib")


        # update bandit bidder (exp3???????)
        pass # for now truthful bidding


###
### CTR regression SGD
###
class NoveltyBidderSGD(NoveltyBidder):
    def __init__(self, rng):
        super(NoveltyBidderSGD, self).__init__(rng, isContinuous=True)

        self.random_state = rng.choice(100)
        self.conv_regressor = SGDRegressor(random_state=self.random_state).fit([[0., 0., 0., 0., 0., 1.]], [0.5])
    
    def bid(self, value, context, estimated_CTR):
        conv = self.conv_regressor.predict(context.reshape(1,-1))[0]
        return conv * value # TODO choose bid more wisely!!
    
    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        super().update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name)

        # update regression model
        # i shouldnt use the whole dataset, but only when i win
        X = contexts[won_mask]
        y = outcomes[won_mask]
        if X.size > 0:
            # X = X.reshape(-1, 1) -> X has already 2 dims
            # y = y.reshape(-1, 1)
            self.conv_regressor.partial_fit(X, y)
            
        # update bandit bidder
        pass # for now truthful bidding

        if iteration == self.num_iterations-1:
            ts = time.strftime("%Y%m%d-%H%M", time.localtime())
            joblib.dump(self.conv_regressor, "./models/sgd"+ts+".joblib")


###
### CTR regression Neural Network
###
import torch as th
class NoveltyBidderNN(NoveltyBidder):
    def __init__(self, rng, epochs=256, device=None, pretrained_model=None):
        super(NoveltyBidderNN, self).__init__(rng, isContinuous=True)

        self.random_state = rng.choice(100)
        if pretrained_model is not None:
            self.conv_regressor = th.load(pretrained_model)
        else:
            self.conv_regressor = th.nn.Sequential(
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
        self.optimizer = th.optim.Adam(self.conv_regressor.parameters(), lr=0.001)
        self.epochs = epochs

        self.contexts = []
        self.outcomes = []

    def bid(self, value, context, estimated_CTR):
        conv = self.conv_regressor(th.tensor(context, dtype=th.float32)).item()
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

            self.conv_regressor.to(self.device)
            self.conv_regressor.train()
            
            for i in range(self.epochs):
                self.optimizer.zero_grad()
                y_pred = self.conv_regressor(X)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optimizer.step()

            if iteration == self.num_iterations-1:
                ts = time.strftime("%Y%m%d-%H%M", time.localtime())
                th.save(self.conv_regressor, "./models/nn_6-4-2-1_"+ts+".pt")