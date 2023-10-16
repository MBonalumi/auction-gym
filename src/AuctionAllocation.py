import numpy as np

class AllocationMechanism:
    ''' Base class for allocation mechanisms '''
    def __init__(self):
        pass

    def allocate(self, bids, num_slots):
        pass


class FirstPrice(AllocationMechanism):
    ''' (Generalised) First-Price Allocation '''

    def __init__(self):
        super(FirstPrice, self).__init__()

    def allocate(self, bids, num_slots):
        bids1 = np.copy(bids)
        winners = np.argsort(-bids1)[:num_slots]

        # NOTE: ADDED BY ME TO BREAKS TIES RANDOMLY
        winning_bids = np.sort( np.unique(bids1[winners]) )[::-1]
        winning_bids_indices = [np.where(bids1 == bid)[0] for bid in winning_bids]
        for indices in winning_bids_indices:
            np.random.shuffle(indices)
        winners_shuffled = np.concatenate(winning_bids_indices)[:num_slots]

        sorted_bids = -np.sort(-bids1)
        prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        return winners_shuffled, prices, second_prices


class SecondPrice(AllocationMechanism):
    ''' (Generalised) Second-Price Allocation '''

    def __init__(self):
        super(SecondPrice, self).__init__()

    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        prices = -np.sort(-bids)[1:num_slots+1]
        return winners, prices, prices
