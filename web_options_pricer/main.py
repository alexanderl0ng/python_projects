import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
from option_pricing_models import *

option_pricing_method = st.sidebar.selectbox("Which pricing model?", ("Black-Scholes", "Binomial", "Monte Carlo"))

S = st.sidebar.number_input('Spot Price:', min_value=0., value=100., step=1., format="%.2f")
K = st.sidebar.number_input('Strike Price', min_value=0., value=100., step=1., format="%.2f")
time_to_maturity = st.sidebar.number_input('Time to Maturity (Days)', min_value=1, value=365) / 365
risk_free_rate = st.sidebar.number_input('Risk Free Rate (%)', min_value=0., value=5., step=1., format="%.2f") / 100
volatility = st.sidebar.number_input('Volatility (%)', min_value=0., value=20., step=1., format="%.2f") / 100
option_type = st.sidebar.selectbox('Option Type:', ("Call", "Put"))
col1, col2 = st.sidebar.columns(2)
dividend = col1.toggle('Dividend')
dividend_rate = 0

if dividend:
	dividend_rate = col2.number_input('Annualised Dividend Rate (%)', min_value=0., step=1., format="%.2f") / 100

# if dividend:
# 	fixed_amount = col1.checkbox('Fixed Amount')

# if dividend and not fixed_amount:
# 	dividend_rate = col2.number_input('Annualised Dividend Rate (%)', min_value=0., step=1., format="%.2f")
# elif dividend and fixed_amount:
# 	dividend_rate = col2.number_input('Fixed Amount', min_value=0., step=1., format="%.2f")
# 	ex_div_date = col2.number_input('Days to Ex-Dividend', min_value=0, step=1)

option = Option(S, K, time_to_maturity, risk_free_rate, volatility, dividend_rate, option_type)

if (option_pricing_method == "Monte Carlo"):
	st.title("Monte Carlo")

	simulations = st.sidebar.number_input('Number of Simulations', min_value=100, max_value=100000, value=1000)

	mc = MonteCarloModel(option, simulations)

	mc_col1, mc_col2 = st.columns(2)
	mc_col2.metric('Option Value', mc.price())

	mc_col3, mc_col4 = st.columns(2)

	min_spot = mc_col3.number_input('Min. Spot Price', min_value=0., value=option.spot_price * 0.75, step=1.)
	min_vol = mc_col4.slider('Min. Volatility', min_value=0., max_value=100., value=(option.volatility * 0.5) * 100) / 100

	mc_col5, mc_col6 = st.columns(2)

	max_spot = mc_col5.number_input('Max. Spot Price', min_value=0., value=option.spot_price * 1.25, step=1.)
	max_vol= mc_col6.slider('Max. Volatility', min_value=0., max_value=100., value=(option.volatility * 1.5) * 100) / 100

	if (max_vol < min_vol):
		st.warning("Maximum Volatility is less than Minimum Volatility")

	if (max_spot < min_spot):
		st.warning("Maximum Spot Price is less than Minimum Spot Price")

	model_params = {'simulations': simulations}

	sens_analysis = SensitivityAnalysis(option, MonteCarloModel, min_spot, max_spot, min_vol, max_vol, model_params)

	coordinates = sens_analysis.find_coordinates()

	plot = sens_analysis.plot(mc.price())

	st.write(plot)

if (option_pricing_method == "Black-Scholes"):
	st.title("Black-Scholes-Merton (BSM)")

	bsm = BlackScholesModel(option)

	bsm_col1, bsm_col2 = st.columns(2)
	bsm_col2.metric('Option Value', bsm.price())

	bsm_col3, bsm_col4 = st.columns(2)

	min_spot = bsm_col3.number_input('Min. Spot Price', min_value=0., value=option.spot_price * 0.75, step=1.)
	min_vol = bsm_col4.slider('Min. Volatility', min_value=0., max_value=100., value=(option.volatility * 0.5) * 100) / 100

	bsm_col5, bsm_col6 = st.columns(2)

	max_spot = bsm_col5.number_input('Max. Spot Price', min_value=0., value=option.spot_price * 1.25, step=1.)
	max_vol= bsm_col6.slider('Max. Volatility', min_value=0., max_value=100., value=(option.volatility * 1.5) * 100) / 100

	if (max_vol < min_vol):
		st.warning("Maximum Volatility is less than Minimum Volatility")

	if (max_spot < min_spot):
		st.warning("Maximum Spot Price is less than Minimum Spot Price")

	sens_analysis = SensitivityAnalysis(option, BlackScholesModel, min_spot, max_spot, min_vol, max_vol)

	coordinates = sens_analysis.find_coordinates()

	plot = sens_analysis.plot(bsm.price())

	st.write(plot)

if (option_pricing_method == "Binomial"):
	st.title("Binomial Options Pricing Model (BOPM)")

	steps = st.sidebar.number_input('Number of Steps', min_value=1, value=6)

	btm = BinomialTreeModel(option, steps)

	bopm_col1, bopm_col2 = st.columns(2)
	bopm_col2.metric('Option Value', btm.price())

	st.subheader("This model uses the Cox-Ross-Rubinstein binomial model.")

	bopm_col3, bopm_col4 = st.columns(2)

	min_spot = bopm_col3.number_input('Min. Spot Price', min_value=0., value=option.spot_price * 0.75, step=1.)
	min_vol = bopm_col4.slider('Min. Volatility', min_value=0., max_value=100., value=(option.volatility * 0.5) * 100) / 100

	bopm_col5, bopm_col6 = st.columns(2)

	max_spot = bopm_col5.number_input('Max. Spot Price', min_value=0., value=option.spot_price * 1.25, step=1.)
	max_vol= bopm_col6.slider('Max. Volatility', min_value=0., max_value=100., value=(option.volatility * 1.5) * 100) / 100

	if (max_vol < min_vol):
		st.warning("Maximum Volatility is less than Minimum Volatility")

	if (max_spot < min_spot):
		st.warning("Maximum Spot Price is less than Minimum Spot Price")

	model_params = {'steps': steps}

	sens_analysis = SensitivityAnalysis(option, BinomialTreeModel, min_spot, max_spot, min_vol, max_vol, model_params)

	coordinates = sens_analysis.find_coordinates()

	plot = sens_analysis.plot(btm.price())

	st.write(plot)




