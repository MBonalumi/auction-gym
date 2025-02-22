from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np

from BidderAllocation import OracleAllocator
from Models import sigmoid

from utils import is_ctr_loosen as CTR_LOOSEN, is_discretized as DISCRETIZED
from utils import scaleup_ctr as SCALEUP_CTR

# DISCRETIZED =  DISCRETIZED()
# CTR_LOOSEN = CTR_LOOSEN()

#######
#######
####### RICORDA DI MODIFICARE BidderAllocation , oltre a questo file
#######
#######

class Auction:
    ''' Base class for auctions '''
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, embedding_size, embedding_var, obs_embedding_size, num_participants_per_round):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0

        self.agent2items = agent2items
        self.agents2item_values = agents2item_values

        self.embedding_size = embedding_size
        self.embedding_var = embedding_var

        self.obs_embedding_size = obs_embedding_size

        self.num_participants_per_round = num_participants_per_round

        # NEW ADDED
        self.DISCRETIZED = DISCRETIZED()
        self.CTR_LOOSEN = CTR_LOOSEN()

        print(f"configuration: discretized? {self.DISCRETIZED}, ctr-loosen? {self.CTR_LOOSEN}")

    def simulate_opportunity(self):
        # Sample the number of slots uniformly between [1, max_slots]
        num_slots = self.rng.integers(1, self.max_slots + 1)

        # Sample a true context vector  TODO: discretized true context
        true_context = np.concatenate((self.rng.normal(0, self.embedding_var, size=self.embedding_size), [1.0]))

        if self.DISCRETIZED:
            # Discretize true context
            discrete_space = np.array([-1.09, 0.0, 1.09])  # centroids of a gaussian divided in 3, found through analysis in `Testing Stuff.ipynb`
            bin_separator = np.array([-0.4307, 0.4307])

            true_context = discrete_space[np.digitize(true_context[:self.embedding_size], bins=bin_separator)]
            true_context = np.concatenate((true_context, [1.0])) 

        # Mask true context into observable context
        obs_context = np.concatenate((true_context[:self.obs_embedding_size], [1.0]))

        true_context = true_context.astype(np.float32)
        obs_context = obs_context.astype(np.float32)

        # At this point, the auctioneer solicits bids from
        # the list of bidders that might want to compete.
        bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]
        for agent in participating_agents:
            # Get the bid and the allocated item
            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            else:
                bid, item = agent.bid(obs_context)
            bids.append(bid)
            # Compute the true CTRs for items in this agent's catalogue TODO: modified CTR calculation
            true_CTR = sigmoid(true_context @ self.agent2items[agent.name].T)
            
            if self.CTR_LOOSEN:
                # true_CTR = sigmoid(true_context @ self.agent2items[agent.name].T) * 0.7 + 0.3     # leaving last dimension (discreases), but scaling up ctr
                true_CTR = sigmoid(true_context[:-1] @ self.agent2items[agent.name].T[:-1])    # loosen ctr, remove last dimension to increase values
                true_CTR = SCALEUP_CTR(true_CTR)     # scaling up the ctr, implemented in utils.py
            
            agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
            CTRs.append(true_CTR[item])
        bids = np.array(bids, dtype=object)
        CTRs = np.array(CTRs)

        # Now we have bids, we need to somehow allocate slots
        # "second_prices" tell us how much lower the winner could have gone without changing the outcome
        winners, prices, second_prices = self.allocation.allocate(bids, num_slots)

        # Bidders only obtain value when they get their outcome
        # Either P(view), P(click | view, ad), P(conversion | click, view, ad)
        # For now, look at P(click | ad) * P(view)
        outcomes = self.rng.binomial(1, CTRs[winners])

        # Let bidders know what they're being charged for
        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, second_price, bool(outcome))
                else:
                    agent.set_price(price)
            self.revenue += price
        
        return participating_agents_idx, bids     #added to keep track + calculate regret

    def clear_revenue(self):
        self.revenue = 0.0
