"""
Financial time series generators.

This module provides generators for realistic financial data including stock prices,
returns, volatility patterns, interest rates, and multi-asset correlations.

**Author**: Dr. Saad Laouadi
**Email**: dr.saad.laouadi@gmail.com
**Project**: timeseries-hub
**License**: MIT
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Union, Tuple, Dict, Any
from scipy.stats import norm
from .base import TimeSeriesGenerator


class GeometricBrownianMotionGenerator(TimeSeriesGenerator):
	"""
	Generate stock prices using Geometric Brownian Motion (Black-Scholes model).

	The classic model for stock price movements:
	dS = μ*S*dt + σ*S*dW

	Where S is stock price, μ is drift (expected return), σ is volatility,
	and dW is a Brownian motion increment.

	Parameters
	----------
	initial_price : float, default 100.0
		Starting stock price
	drift : float, default 0.08
		Annual expected return (8% = 0.08)
	volatility : float, default 0.20
		Annual volatility (20% = 0.20)
	time_step : float, default 1/252
		Time step in years (daily = 1/252)
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Large cap stock (moderate volatility)
	>>> stock_gen = GeometricBrownianMotionGenerator(
	...     initial_price=150.0,
	...     drift=0.10,      # 10% annual return
	...     volatility=0.25,  # 25% annual volatility
	...     time_step=1/252   # Daily data
	... )
	>>> prices = stock_gen.generate(n_points=252)  # 1 year
	>>>
	>>> # High-growth tech stock
	>>> tech_gen = GeometricBrownianMotionGenerator(
	...     initial_price=50.0,
	...     drift=0.15,      # 15% annual return
	...     volatility=0.40   # 40% annual volatility
	... )
	>>> tech_prices = tech_gen.generate(n_points=1260)  # 5 years
	"""
	
	def __init__(self, initial_price: float = 100.0, drift: float = 0.08, volatility: float = 0.20,
	             time_step: float = 1 / 252, seed: Optional[int] = None):
		super().__init__(seed)
		self.initial_price = initial_price
		self.drift = drift
		self.volatility = volatility
		self.time_step = time_step
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D',
	             return_log_returns: bool = False, **kwargs) -> pd.Series:
		"""
		Generate stock price series using GBM.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string (D=daily recommended for financial data)
		return_log_returns : bool, default False
			If True, return log returns instead of prices
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series
			Stock price series or log returns series
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate random increments
		dt = self.time_step
		random_increments = self.rng.normal(0, 1, n_points)
		
		# Calculate log returns using GBM formula
		log_returns = (self.drift - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * random_increments
		
		if return_log_returns:
			index = self._create_datetime_index(n_points, start_date, freq)
			return pd.Series(log_returns, index=index, name='log_returns')
		
		# Convert to prices
		cumulative_log_returns = np.cumsum(log_returns)
		prices = self.initial_price * np.exp(cumulative_log_returns)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(prices, index=index, name='stock_price')


class MeanRevertingGenerator(TimeSeriesGenerator):
	"""
	Generate mean-reverting time series using Ornstein-Uhlenbeck process.

	The process: dX = θ(μ - X)dt + σdW

	Where θ is mean reversion speed, μ is long-term mean, σ is volatility.
	Common for interest rates, commodity prices, and volatility.

	Parameters
	----------
	initial_value : float, default 0.05
		Starting value
	long_term_mean : float, default 0.05
		Long-term mean the process reverts to
	reversion_speed : float, default 2.0
		Speed of mean reversion (higher = faster reversion)
	volatility : float, default 0.02
		Volatility of the process
	time_step : float, default 1/252
		Time step in years
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Interest rate model
	>>> rate_gen = MeanRevertingGenerator(
	...     initial_value=0.03,     # 3% starting rate
	...     long_term_mean=0.04,    # 4% long-term rate
	...     reversion_speed=1.5,    # Moderate reversion
	...     volatility=0.015        # 1.5% volatility
	... )
	>>> rates = rate_gen.generate(n_points=1260)  # 5 years
	>>>
	>>> # Commodity price (oil, gold)
	>>> commodity_gen = MeanRevertingGenerator(
	...     initial_value=75.0,     # $75/barrel oil
	...     long_term_mean=80.0,    # $80 long-term
	...     reversion_speed=0.8,    # Slow reversion
	...     volatility=8.0          # High volatility
	... )
	>>> oil_prices = commodity_gen.generate(n_points=252)
	"""
	
	def __init__(self, initial_value: float = 0.05, long_term_mean: float = 0.05, reversion_speed: float = 2.0,
	             volatility: float = 0.02, time_step: float = 1 / 252, seed: Optional[int] = None):
		super().__init__(seed)
		self.initial_value = initial_value
		self.long_term_mean = long_term_mean
		self.reversion_speed = reversion_speed
		self.volatility = volatility
		self.time_step = time_step
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate mean-reverting time series.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series
			Mean-reverting time series
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		dt = self.time_step
		series = np.zeros(n_points)
		series[0] = self.initial_value
		
		# Generate Ornstein-Uhlenbeck process
		for i in range(1, n_points):
			dW = self.rng.normal(0, np.sqrt(dt))
			drift = self.reversion_speed * (self.long_term_mean - series[i - 1]) * dt
			diffusion = self.volatility * dW
			series[i] = series[i - 1] + drift + diffusion
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(series, index=index, name='mean_reverting')


class JumpDiffusionGenerator(TimeSeriesGenerator):
	"""
	Generate stock prices with jumps using Merton Jump Diffusion model.

	Combines geometric Brownian motion with occasional large jumps,
	capturing market crashes and sudden price movements.

	Parameters
	----------
	initial_price : float, default 100.0
		Starting stock price
	drift : float, default 0.08
		Annual expected return (excluding jumps)
	volatility : float, default 0.20
		Annual diffusion volatility
	jump_intensity : float, default 0.1
		Average number of jumps per year
	jump_mean : float, default -0.02
		Average jump size (log scale, negative for crashes)
	jump_volatility : float, default 0.08
		Volatility of jump sizes
	time_step : float, default 1/252
		Time step in years
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Stock with crash risk
	>>> crash_stock = JumpDiffusionGenerator(
	...     initial_price=100.0,
	...     drift=0.12,
	...     volatility=0.25,
	...     jump_intensity=0.2,     # 0.2 jumps per year on average
	...     jump_mean=-0.15,        # -15% average crash
	...     jump_volatility=0.25    # Variable crash sizes
	... )
	>>> risky_prices = crash_stock.generate(n_points=1260)  # 5 years
	>>>
	>>> # Crypto-like with both crashes and pumps
	>>> crypto_gen = JumpDiffusionGenerator(
	...     initial_price=50000,
	...     drift=0.20,
	...     volatility=0.60,
	...     jump_intensity=1.0,     # 1 jump per year
	...     jump_mean=0.0,          # Neutral average
	...     jump_volatility=0.40    # Large variable jumps
	... )
	>>> crypto_prices = crypto_gen.generate(n_points=365)
	"""
	
	def __init__(self, initial_price: float = 100.0, drift: float = 0.08, volatility: float = 0.20,
	             jump_intensity: float = 0.1, jump_mean: float = -0.02, jump_volatility: float = 0.08,
	             time_step: float = 1 / 252, seed: Optional[int] = None):
		super().__init__(seed)
		self.initial_price = initial_price
		self.drift = drift
		self.volatility = volatility
		self.jump_intensity = jump_intensity
		self.jump_mean = jump_mean
		self.jump_volatility = jump_volatility
		self.time_step = time_step
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate jump diffusion stock prices.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series
			Stock price series with jumps
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		dt = self.time_step
		
		# Generate diffusion component (GBM)
		random_increments = self.rng.normal(0, 1, n_points)
		diffusion_returns = (self.drift - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(
				dt) * random_increments
		
		# Generate jumps
		jump_times = self.rng.poisson(self.jump_intensity * dt, n_points)
		jump_returns = np.zeros(n_points)
		
		for i in range(n_points):
			if jump_times[i] > 0:
				# Multiple jumps possible in one period (rare)
				total_jump = 0
				for _ in range(jump_times[i]):
					jump_size = self.rng.normal(self.jump_mean, self.jump_volatility)
					total_jump += jump_size
				jump_returns[i] = total_jump
		
		# Combine diffusion and jumps
		total_log_returns = diffusion_returns + jump_returns
		cumulative_log_returns = np.cumsum(total_log_returns)
		prices = self.initial_price * np.exp(cumulative_log_returns)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(prices, index=index, name='jump_diffusion_price')


class GARCHGenerator(TimeSeriesGenerator):
	"""
	Generate returns with GARCH(1,1) volatility clustering.

	Models the empirical fact that periods of high volatility tend to be
	followed by periods of high volatility (volatility clustering).

	Returns: r_t = σ_t * ε_t
	Volatility: σ_t² = ω + α*r_{t-1}² + β*σ_{t-1}²

	Parameters
	----------
	omega : float, default 0.00001
		Baseline variance level
	alpha : float, default 0.1
		Reaction to market shocks (ARCH effect)
	beta : float, default 0.85
		Persistence of volatility (GARCH effect)
	initial_volatility : float, default 0.02
		Starting volatility level (daily)
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Typical stock return volatility clustering
	>>> garch_gen = GARCHGenerator(
	...     omega=0.000002,  # Low baseline
	...     alpha=0.12,      # Moderate shock reaction
	...     beta=0.85,       # High persistence
	...     initial_volatility=0.015  # 1.5% daily vol
	... )
	>>> returns = garch_gen.generate(n_points=1260)  # 5 years
	>>>
	>>> # High volatility market (emerging markets, crypto)
	>>> volatile_gen = GARCHGenerator(
	...     omega=0.00001,
	...     alpha=0.20,      # High shock reaction
	...     beta=0.75,       # Moderate persistence
	...     initial_volatility=0.03  # 3% daily vol
	... )
	>>> volatile_returns = volatile_gen.generate(n_points=252)
	"""
	
	def __init__(self, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85, initial_volatility: float = 0.02,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.omega = omega
		self.alpha = alpha
		self.beta = beta
		self.initial_volatility = initial_volatility
		
		# Check GARCH stability condition
		if alpha + beta >= 1:
			import warnings
			warnings.warn("GARCH parameters may lead to explosive volatility", UserWarning)
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D',
	             return_volatility: bool = False, **kwargs) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
		"""
		Generate GARCH returns with volatility clustering.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		return_volatility : bool, default False
			If True, return both returns and volatility series
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series or tuple
			Returns series, or (returns, volatility) if return_volatility=True
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		returns = np.zeros(n_points)
		volatility = np.zeros(n_points)
		volatility[0] = self.initial_volatility
		
		# Generate standardized innovations
		innovations = self.rng.normal(0, 1, n_points)
		
		# Generate GARCH process
		for t in range(n_points):
			returns[t] = volatility[t] * innovations[t]
			
			if t < n_points - 1:
				# Update volatility for next period
				volatility[t + 1] = np.sqrt(self.omega + self.alpha * returns[t] ** 2 + self.beta * volatility[t] ** 2)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		returns_series = pd.Series(returns, index=index, name='garch_returns')
		
		if return_volatility:
			vol_series = pd.Series(volatility, index=index, name='volatility')
			return returns_series, vol_series
		
		return returns_series


class MultiAssetGenerator(TimeSeriesGenerator):
	"""
	Generate correlated multi-asset returns.

	Creates realistic correlations between multiple assets using
	Cholesky decomposition of the correlation matrix.

	Parameters
	----------
	n_assets : int, default 3
		Number of assets to generate
	correlation_matrix : np.ndarray, optional
		Correlation matrix (n_assets x n_assets). If None, uses random correlations.
	volatilities : list of float, optional
		Individual asset volatilities. If None, uses random volatilities.
	drifts : list of float, optional
		Individual asset drifts (expected returns). If None, uses random drifts.
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Three-asset portfolio (stocks, bonds, commodities)
	>>> correlation_matrix = np.array([
	...     [1.0, 0.3, 0.1],    # Stock correlations
	...     [0.3, 1.0, -0.2],   # Bond correlations
	...     [0.1, -0.2, 1.0]    # Commodity correlations
	... ])
	>>>
	>>> multi_gen = MultiAssetGenerator(
	...     n_assets=3,
	...     correlation_matrix=correlation_matrix,
	...     volatilities=[0.20, 0.05, 0.30],  # Stock, bond, commodity vols
	...     drifts=[0.08, 0.03, 0.06]         # Expected returns
	... )
	>>> returns_df = multi_gen.generate(n_points=252)
	>>>
	>>> # Random 5-asset portfolio
	>>> random_gen = MultiAssetGenerator(n_assets=5)
	>>> random_returns = random_gen.generate(n_points=1260)
	"""
	
	def __init__(self, n_assets: int = 3, correlation_matrix: Optional[np.ndarray] = None,
	             volatilities: Optional[List[float]] = None, drifts: Optional[List[float]] = None,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.n_assets = n_assets
		
		# Set default correlation matrix
		if correlation_matrix is None:
			# Generate random correlation matrix
			A = self.rng.randn(n_assets, n_assets)
			correlation_matrix = np.corrcoef(A)
		
		# Validate correlation matrix
		if correlation_matrix.shape != (n_assets, n_assets):
			raise ValueError("Correlation matrix must be n_assets x n_assets")
		
		# Check positive semi-definite
		eigenvals = np.linalg.eigvals(correlation_matrix)
		if np.any(eigenvals < -1e-8):
			raise ValueError("Correlation matrix must be positive semi-definite")
		
		self.correlation_matrix = correlation_matrix
		
		# Set default volatilities and drifts
		if volatilities is None:
			volatilities = 0.15 + 0.15 * self.rng.rand(n_assets)  # 15%-30%
		if drifts is None:
			drifts = 0.05 + 0.10 * self.rng.rand(n_assets)  # 5%-15%
		
		self.volatilities = np.array(volatilities)
		self.drifts = np.array(drifts)
		
		# Compute Cholesky decomposition for correlation
		try:
			self.cholesky = np.linalg.cholesky(correlation_matrix)
		except np.linalg.LinAlgError:
			# Use eigenvalue decomposition if Cholesky fails
			eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
			eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
			self.cholesky = eigenvecs @ np.diag(np.sqrt(eigenvals))
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', return_prices: bool = False,
	             initial_prices: Optional[List[float]] = None, **kwargs) -> pd.DataFrame:
		"""
		Generate multi-asset returns or prices.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		return_prices : bool, default False
			If True, return cumulative prices instead of returns
		initial_prices : list of float, optional
			Starting prices for each asset (only used if return_prices=True)
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.DataFrame
			Multi-asset returns or prices with assets as columns
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate independent standard normal innovations
		independent_innovations = self.rng.normal(0, 1, (n_points, self.n_assets))
		
		# Apply correlation structure
		correlated_innovations = independent_innovations @ self.cholesky.T
		
		# Scale by volatilities and add drifts
		dt = 1 / 252  # Daily time step
		returns = np.zeros((n_points, self.n_assets))
		
		for i in range(self.n_assets):
			returns[:, i] = (self.drifts[i] * dt + self.volatilities[i] * np.sqrt(dt) * correlated_innovations[:, i])
		
		# Create DataFrame
		index = self._create_datetime_index(n_points, start_date, freq)
		asset_names = [f'Asset_{i + 1}' for i in range(self.n_assets)]
		returns_df = pd.DataFrame(returns, index=index, columns=asset_names)
		
		if return_prices:
			if initial_prices is None:
				initial_prices = [100.0] * self.n_assets
			
			# Convert returns to prices
			log_returns = returns_df.values
			cumulative_log_returns = np.cumsum(log_returns, axis=0)
			prices = np.array(initial_prices) * np.exp(cumulative_log_returns)
			
			return pd.DataFrame(prices, index=index, columns=asset_names)
		
		return returns_df


class CryptocurrencyGenerator(TimeSeriesGenerator):
	"""
	Generate cryptocurrency-like time series with extreme volatility patterns.

	Combines high volatility, frequent jumps, and momentum effects
	typical of cryptocurrency markets.

	Parameters
	----------
	initial_price : float, default 50000.0
		Starting price (e.g., Bitcoin ~$50k)
	base_volatility : float, default 0.04
		Base daily volatility (4% = very high)
	jump_intensity : float, default 0.5
		Frequency of large price jumps
	momentum_factor : float, default 0.02
		Strength of momentum/trend following
	crash_probability : float, default 0.02
		Probability of major crashes
	pump_probability : float, default 0.015
		Probability of major pumps
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Bitcoin-like cryptocurrency
	>>> bitcoin_gen = CryptocurrencyGenerator(
	...     initial_price=45000,
	...     base_volatility=0.035,
	...     jump_intensity=0.3,
	...     momentum_factor=0.025,
	...     crash_probability=0.01,
	...     pump_probability=0.008
	... )
	>>> btc_prices = bitcoin_gen.generate(n_points=365)
	>>>
	>>> # Altcoin (more volatile)
	>>> altcoin_gen = CryptocurrencyGenerator(
	...     initial_price=100,
	...     base_volatility=0.08,     # 8% daily volatility
	...     jump_intensity=1.0,       # Frequent jumps
	...     momentum_factor=0.04,     # Strong momentum
	...     crash_probability=0.03,   # Frequent crashes
	...     pump_probability=0.03     # Frequent pumps
	... )
	>>> altcoin_prices = altcoin_gen.generate(n_points=180)
	"""
	
	def __init__(self, initial_price: float = 50000.0, base_volatility: float = 0.04, jump_intensity: float = 0.5,
	             momentum_factor: float = 0.02, crash_probability: float = 0.02, pump_probability: float = 0.015,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.initial_price = initial_price
		self.base_volatility = base_volatility
		self.jump_intensity = jump_intensity
		self.momentum_factor = momentum_factor
		self.crash_probability = crash_probability
		self.pump_probability = pump_probability
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate cryptocurrency price series.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series
			Cryptocurrency price series
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		log_returns = np.zeros(n_points)
		
		for t in range(n_points):
			# Base random return
			base_return = self.rng.normal(0, self.base_volatility)
			
			# Add momentum (trend following)
			if t > 0:
				momentum = self.momentum_factor * log_returns[t - 1]
				base_return += momentum
			
			# Random jumps
			if self.rng.random() < self.jump_intensity / 365:
				jump_size = self.rng.normal(0, 0.15)  # Large jumps
				base_return += jump_size
			
			# Major crashes
			if self.rng.random() < self.crash_probability / 365:
				crash_size = -abs(self.rng.normal(0.20, 0.10))  # 20% average crash
				base_return += crash_size
			
			# Major pumps
			if self.rng.random() < self.pump_probability / 365:
				pump_size = abs(self.rng.normal(0.25, 0.15))  # 25% average pump
				base_return += pump_size
			
			log_returns[t] = base_return
		
		# Convert to prices
		cumulative_log_returns = np.cumsum(log_returns)
		prices = self.initial_price * np.exp(cumulative_log_returns)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(prices, index=index, name='crypto_price')


class BondYieldGenerator(TimeSeriesGenerator):
	"""
	Generate realistic government bond yield curves and time series.

	Models interest rate dynamics with mean reversion, term structure,
	and economic cycle effects.

	Parameters
	----------
	initial_yields : dict, optional
		Initial yields by maturity {'1Y': 0.02, '5Y': 0.025, '10Y': 0.03}
	mean_reversion_speed : float, default 0.5
		Speed of mean reversion for interest rates
	yield_volatility : float, default 0.015
		Volatility of yield changes
	economic_cycle_period : int, default 2520
		Length of economic cycle in trading days (~10 years)
	economic_cycle_amplitude : float, default 0.01
		Amplitude of economic cycle effects on yields
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # US Treasury yield curve
	>>> treasury_gen = BondYieldGenerator(
	...     initial_yields={'2Y': 0.025, '5Y': 0.030, '10Y': 0.035, '30Y': 0.040},
	...     mean_reversion_speed=0.3,
	...     yield_volatility=0.012,
	...     economic_cycle_period=2520
	... )
	>>> yield_curves = treasury_gen.generate(n_points=1260)  # 5 years
	>>>
	>>> # Emerging market bonds (higher volatility)
	>>> em_gen = BondYieldGenerator(
	...     initial_yields={'5Y': 0.08, '10Y': 0.09},
	...     yield_volatility=0.025,     # Higher volatility
	...     mean_reversion_speed=0.8    # Faster mean reversion
	... )
	>>> em_yields = em_gen.generate(n_points=252)
	"""
	
	def __init__(self, initial_yields: Optional[Dict[str, float]] = None, mean_reversion_speed: float = 0.5,
	             yield_volatility: float = 0.015, economic_cycle_period: int = 2520,
	             economic_cycle_amplitude: float = 0.01, seed: Optional[int] = None):
		super().__init__(seed)
		
		# Default yield curve
		if initial_yields is None:
			initial_yields = {
					'1Y': 0.020, '2Y': 0.025, '5Y': 0.030, '10Y': 0.035, '30Y': 0.040
					}
		
		self.initial_yields = initial_yields
		self.maturities = list(initial_yields.keys())
		self.mean_reversion_speed = mean_reversion_speed
		self.yield_volatility = yield_volatility
		self.economic_cycle_period = economic_cycle_period
		self.economic_cycle_amplitude = economic_cycle_amplitude
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.DataFrame:
		"""
		Generate bond yield time series.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.DataFrame
			Bond yields with maturities as columns
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		n_maturities = len(self.maturities)
		yields = np.zeros((n_points, n_maturities))
		
		# Set initial yields
		for i, maturity in enumerate(self.maturities):
			yields[0, i] = self.initial_yields[maturity]
		
		# Generate correlated innovations for different maturities
		correlation_matrix = np.exp(-0.1 * np.abs(np.subtract.outer(range(n_maturities), range(n_maturities))))
		cholesky = np.linalg.cholesky(correlation_matrix)
		
		dt = 1 / 252  # Daily time step
		
		for t in range(1, n_points):
			# Economic cycle component
			cycle_effect = (self.economic_cycle_amplitude * np.sin(2 * np.pi * t / self.economic_cycle_period))
			
			# Generate correlated shocks
			independent_shocks = self.rng.normal(0, 1, n_maturities)
			correlated_shocks = cholesky @ independent_shocks
			
			for i in range(n_maturities):
				# Mean reversion to initial yield + cycle
				long_term_mean = self.initial_yields[self.maturities[i]] + cycle_effect
				mean_reversion = self.mean_reversion_speed * (long_term_mean - yields[t - 1, i]) * dt
				
				# Volatility shock
				shock = self.yield_volatility * np.sqrt(dt) * correlated_shocks[i]
				
				# Update yield
				yields[t, i] = yields[t - 1, i] + mean_reversion + shock
				
				# Ensure yields don't go negative
				yields[t, i] = max(yields[t, i], 0.001)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.DataFrame(yields, index=index, columns=self.maturities)