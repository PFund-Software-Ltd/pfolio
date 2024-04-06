from pfund.strategies.strategy_base import BaseStrategy


class Portfolio:
    def __init__(self, universe, positions, balances, preference=None):
        self.universe = universe
        self.positions = positions
        self.balances = balances
        self.preference = preference
        # TODO:
        # if preference is None:
        #     self.logger.warning('No preference provided.')
    
    def rebalance(self, weights: dict[str, float] | None=None, strategy: BaseStrategy | None=None):
        if weights:
            # TODO: rebalances current portfolio based on the weights provided
            pass
        elif strategy:
            strategy.rebalance()
        else:
            raise ValueError('Either weights or strategy must be provided.')
    
    def allocate(self, strategy: BaseStrategy):
        strategy.allocate()
    
    def diversify(self, strategy: BaseStrategy):
        strategy.diversify()
    
    def optimize(self, optimizer):
        pass
