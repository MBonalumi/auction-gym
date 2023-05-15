import numpy as np
from Bidder import Bidder
from BidderBandits import UCB1

################################
######       linUCB       ######
################################
class linUCB(Bidder):
    def __init__(self, rng, feature_dim, gamma=1.0, alpha=1.0):
        super(linUCB, self).__init__(rng)
        self.gamma = gamma
        self.BIDS = [0.005, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.9, 2.4]
        self.NUM_BIDS = len(self.BIDS)
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.A = [np.eye(feature_dim) for _ in range(self.NUM_BIDS)]
        self.b = [np.zeros((feature_dim, 1)) for _ in range(self.NUM_BIDS)]

    def bid(self, value, context, estimated_CTR):
        # context_var = context[-1]
        # context = context[:-1]
        
        # trying to use (value,estimated_CTR) as context
        context = np.array((value, estimated_CTR))

        p = np.zeros(self.NUM_BIDS)
        theta = [np.linalg.inv(self.A[a]).dot(self.b[a]) for a in range(self.NUM_BIDS)]
        for a in range(self.NUM_BIDS):
            p[a] = theta[a].T.dot(context) + self.alpha * np.sqrt(context.T.dot(np.linalg.inv(self.A[a])).dot(context))
        chosen_bid = self.BIDS[np.argmax(p)]
        return chosen_bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # contexts_var = contexts[:, -1]
        # contexts = contexts[:, :-1]

        # trying to use (value,estimated_CTR) as context
        contexts = np.array([(v,ctr) for (v,ctr) in zip(values, estimated_CTRs)])

        for i in range(len(bids)):
            action = self.BIDS.index(bids[i])
            self.A[action] += contexts[i].dot(contexts[i].T)
            if won_mask[i]:
                reward = (values[i] * outcomes[i]) - prices[i]
                pass
                # right hand-side shape is (5,)
                # reshape adds exmpty dim to match -> left hand-side shape is (5,1)
                self.b[action] += (reward * contexts[i]).reshape(-1,1)


################################
######  simple regressor  ######
################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
class simpleRegressor(Bidder):
    def __init__(self, rng, gamma=1.0):
        super(linUCB, self).__init__(rng)
        self.gamma = gamma
        self.model = None

    def bid(self, value, context, estimated_CTR):
        if self.model is None:
            return value*estimated_CTR
        # otherwise use model
        return 1.0 

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        
        pass

    def clear_logs(self, memory):
        pass

