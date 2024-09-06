import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

class Option:
	def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, dividend, option_type):
		self.spot_price = spot_price
		self.strike_price = strike_price
		self.time_to_maturity = time_to_maturity
		self.risk_free_rate = risk_free_rate
		self.volatility = volatility
		self.dividend = dividend
		self.option_type = option_type

class BinomialTreeModel:
	def __init__(self, option, steps):
		self.option = option
		self.steps = steps

	def calculate_tree(self):
		S = self.option.spot_price
		K = self.option.strike_price
		T = self.option.time_to_maturity
		r = self.option.risk_free_rate
		sigma = self.option.volatility
		n = self.steps
		dividend = self.option.dividend

		dt = T / n
		u = np.exp(sigma * np.sqrt(dt))
		d = 1 / u
		p = (np.exp((r - dividend) * dt) - d) / (u - d)
		discount = np.exp(-r * dt)

		stock_prices = np.zeros((n + 1, n + 1))
		option_values = np.zeros((n + 1, n + 1))

		for i in range(n + 1):
			for j in range(n + 1):
				stock_prices[j, i] = S * (u ** (i - j)) * (d ** j)

		if self.option.option_type=='Call':
			option_values[:, n] = np.maximum(stock_prices[:, n] - K, 0)
		elif self.option.option_type=='Put':
			option_values[:, n] = np.maximum(K - stock_prices[:, n], 0)

		for i in range(n - 1, -1, -1):
			for j in range(i + 1):
				option_values[j, i] = (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1]) * discount

		return option_values[0][0]

	def price(self):
		return self.calculate_tree()

class BlackScholesModel:
	def __init__(self, option):
		self.option = option

	def calculate_price(self):
		S = self.option.spot_price
		K = self.option.strike_price
		T = self.option.time_to_maturity
		r = self.option.risk_free_rate
		sigma = self.option.volatility
		dividend = self.option.dividend

		d_1 = (np.log(S / K) + T * (r - dividend + (sigma**2 / 2))) / (sigma * np.sqrt(T))
		d_2 = d_1 - (sigma * np.sqrt(T))

		option_value = 0

		if self.option.option_type=='Call':
			option_value = (S * np.exp(-dividend * T) * phi(d_1)) - (K * np.exp(-r * T) * phi(d_2))
		elif self.option.option_type=='Put':
			option_value = (K * np.exp(-r * T) * phi(-d_2)) - (S * np.exp(-dividend * T) * phi(-d_1))

		return option_value

	def price(self):
		return self.calculate_price()

class MonteCarloModel:
	def __init__(self, option, simulations):
		self.option = option
		self.sims = simulations

	def paths(self):
		S = self.option.spot_price
		K = self.option.strike_price
		T = self.option.time_to_maturity
		r = self.option.risk_free_rate
		sigma = self.option.volatility
		dividend = self.option.dividend

		steps = int(T * 504)
		dt = T / steps

		S_T = np.log(S) + np.cumsum(((r - dividend - sigma**2 / 2) * dt +\
							sigma * np.sqrt(dt) * np.random.normal(size=(steps, self.sims))), axis=0)

		return np.exp(S_T)

	def payoffs(self):
		paths = self.paths()

		if self.option.option_type=='Call':
			return np.maximum(paths[-1] - self.option.strike_price, 0)
		elif self.option.option_type=='Put':
			return np.maximum(self.option.strike_price - paths[-1], 0)

	def price(self):
		payoffs = self.payoffs()

		return np.mean(payoffs) * np.exp(-self.option.risk_free_rate * self.option.time_to_maturity)

class SensitivityAnalysis:
	def __init__(self, option, model_class, min_spot, max_spot, min_vol, max_vol, model_params=None):
		self.option = option
		self.model_class = model_class
		self.model_params = model_params if model_params is not None else {}
		self.min_spot = min_spot
		self.max_spot = max_spot
		self.min_vol = min_vol
		self.max_vol = max_vol

	def find_axis(self):
		spot_diff = self.max_spot - self.min_spot
		vol_diff = self.max_vol - self.min_vol
		spot_step = spot_diff / 10
		vol_step = vol_diff / 10

		spot_list = [self.min_spot]
		vol_list = [self.min_vol]

		for i in range(1, 11):
			spot_list.append(self.min_spot + i * spot_step)
			vol_list.append(self.min_vol + i * vol_step)

		return spot_list, vol_list

	def find_coordinates(self):

		spot_list, vol_list = self.find_axis()

		coordinates = []

		for spot in spot_list:
			for vol in vol_list:
				option = Option(spot, self.option.strike_price, self.option.time_to_maturity, self.option.risk_free_rate, vol, self.option.dividend, self.option.option_type)

				model_instance = self.model_class(option, **self.model_params)

				option_price = model_instance.price()

				coordinates.append((spot, vol, option_price))

		return coordinates

	def plot(self, price):

		coordinates = self.find_coordinates()

		dtype = [('Spot Price', 'float'), ('Volatility', 'float'), ('Value', 'float')]

		array = np.array(coordinates, dtype=dtype)

		spots = np.unique(array['Spot Price'])
		vols = np.unique(array['Volatility'])
		vols = vols[::-1]
		per_vol = vols[:] * 100
		round_vol = np.round(per_vol, decimals=2)
		round_spots = np.round(spots, decimals=2)


		price_matrix = np.zeros((len(round_vol), len(spots)))

		for coord in array:
			vol_index = np.where(vols == coord['Volatility'])[0][0]
			spot_index = np.where(spots == coord['Spot Price'])[0][0]
			price_matrix[vol_index, spot_index] = coord['Value']

		fig, ax = plt.subplots()
		cmap = sns.diverging_palette(10, 120, s=100, center="light", as_cmap=True)

		sns.heatmap(price_matrix, ax=ax, center = price, xticklabels=spots, yticklabels=vols, annot=True, fmt=".1f", linewidth=.3, cmap=cmap)

		ax.set_xlabel("Spot Price")
		ax.set_xticklabels(round_spots, rotation=45)
		ax.set_ylabel("Volatility (%)")
		ax.set_yticklabels(round_vol, rotation=0)
		ax.set_title("Option Value for Different Volatilities and Spot Prices")

		return fig






