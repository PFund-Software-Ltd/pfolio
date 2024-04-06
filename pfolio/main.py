from pfolio.universe import Universe
from pfolio.portfolio import Portfolio
from pfolio.preference import Preference


if __name__ == '__main__':
    # TODO:
    positions = {}
    balances = {}
    universe = Universe()
    preference = Preference()
    portfolio = Portfolio(
        universe,
        positions,
        balances,
        preference=preference,
        # TODO:
        # config={
        #     'min_weight': 0.01,
        #     'max_weight': 0.15,
        # }
    )