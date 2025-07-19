"""
Composite time series generators.

This module provides generators that combine multiple components (trends, seasonal
patterns, noise) to create complex, realistic time series patterns.

**Author**: Dr. Saad Laouadi
**Email**: dr.saad.laouadi@gmail.com
**Project**: timeseries-hub
**License**: MIT
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Union, Callable, Dict, Any
from .base import TimeSeriesGenerator
from .trend import (
	LinearTrendGenerator, ExponentialTrendGenerator, PolynomialTrendGenerator
	)
from .seasonal import (
	SeasonalGenerator, MultiSeasonalGenerator, AdditiveSeasonalGenerator
	)
from .noise import (
	WhiteNoiseGenerator, ColoredNoiseGenerator, AutoregressiveNoiseGenerator
	)


class CompositeGenerator(TimeSeriesGenerator):
	"""
	Base class for combining multiple time series generators.

	Provides flexible composition of different generators with various
	combination methods (additive, multiplicative, custom).

	Parameters
	----------
	generators : list of TimeSeriesGenerator
		List of generators to combine
	combination_method : str or callable, default 'additive'
		How to combine the generators:
		- 'additive': sum all components
		- 'multiplicative': multiply all components
		- callable: custom combination function
	weights : list of float, optional
		Weights for each generator. If None, all weights are 1.0
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> from datasets.generators.trend import LinearTrendGenerator
	>>> from datasets.generators.seasonal import SeasonalGenerator
	>>> from datasets.generators.noise import WhiteNoiseGenerator
	>>>
	>>> # Create individual components
	>>> trend = LinearTrendGenerator(slope=0.1)
	>>> seasonal = SeasonalGenerator(period=12, amplitude=2.0)
	>>> noise = WhiteNoiseGenerator(std=0.5)
	>>>
	>>> # Combine additively
	>>> composite = CompositeGenerator(
	...     generators=[trend, seasonal, noise],
	...     combination_method='additive'
	... )
	>>> data = composite.generate(n_points=365)
	"""
	
	def __init__(self, generators: List[TimeSeriesGenerator], combination_method: Union[str, Callable] = 'additive',
	             weights: Optional[List[float]] = None, seed: Optional[int] = None):
		super().__init__(seed)
		
		if not generators:
			raise ValueError("At least one generator must be provided")
		
		self.generators = generators
		self.combination_method = combination_method
		self.weights = weights if weights is not None else [1.0] * len(generators)
		
		if len(self.weights) != len(generators):
			raise ValueError("Number of weights must match number of generators")
		
		# Set same seed for all generators if provided
		if seed is not None:
			for i, gen in enumerate(self.generators):
				gen.set_seed(seed + i)  # Offset seeds to avoid identical sequences
	
	def _combine_components(self, components: List[pd.Series]) -> pd.Series:
		"""Combine multiple time series components."""
		if not components:
			raise ValueError("No components to combine")
		
		# Apply weights
		weighted_components = []
		for component, weight in zip(components, self.weights):
			weighted_components.append(weight * component)
		
		if callable(self.combination_method):
			# Custom combination function
			return self.combination_method(weighted_components)
		elif self.combination_method == 'additive':
			# Sum all components
			result = weighted_components[0].copy()
			for component in weighted_components[1:]:
				result = result + component.values  # Add values, keep first index
			return result
		elif self.combination_method == 'multiplicative':
			# Multiply all components
			result = weighted_components[0].copy()
			for component in weighted_components[1:]:
				result = result * component.values  # Multiply values, keep first index
			return result
		else:
			raise ValueError(f"Unknown combination method: {self.combination_method}")
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate composite time series.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string (D=daily, H=hourly, etc.)
		**kwargs
			Additional keyword arguments passed to individual generators

		Returns
		-------
		pd.Series
			Combined time series
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate all components with same parameters
		components = []
		for generator in self.generators:
			component = generator.generate(n_points, start_date, freq, **kwargs)
			components.append(component)
		
		# Combine components
		result = self._combine_components(components)
		result.name = 'composite'
		
		return result


class TrendSeasonalGenerator(CompositeGenerator):
	"""
	Convenient generator combining trend and seasonal patterns.

	Pre-configured composite generator for the common case of trend + seasonality.

	Parameters
	----------
	trend_type : str, default 'linear'
		Type of trend: 'linear', 'exponential', 'polynomial'
	trend_params : dict, optional
		Parameters for the trend generator
	seasonal_periods : list of int, default [12]
		Periods for seasonal components
	seasonal_amplitudes : list of float, default [1.0]
		Amplitudes for seasonal components
	noise_std : float, default 0.1
		Standard deviation of added noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Simple trend + yearly seasonality
	>>> gen = TrendSeasonalGenerator(
	...     trend_type='linear',
	...     trend_params={'slope': 0.05, 'intercept': 10},
	...     seasonal_periods=[365],
	...     seasonal_amplitudes=[3.0],
	...     noise_std=0.2
	... )
	>>> data = gen.generate(n_points=1095)  # 3 years
	>>>
	>>> # Exponential trend with multi-seasonality
	>>> gen = TrendSeasonalGenerator(
	...     trend_type='exponential',
	...     trend_params={'growth_rate': 0.001},
	...     seasonal_periods=[365, 7],  # Yearly + weekly
	...     seasonal_amplitudes=[2.0, 0.5],
	...     noise_std=0.15
	... )
	>>> data = gen.generate(n_points=730)  # 2 years
	"""
	
	def __init__(self, trend_type: str = 'linear', trend_params: Optional[Dict[str, Any]] = None,
	             seasonal_periods: List[int] = [12], seasonal_amplitudes: List[float] = [1.0], noise_std: float = 0.1,
	             seed: Optional[int] = None):
		
		# Set default parameters
		if trend_params is None:
			trend_params = {}
		
		# Create trend generator
		if trend_type == 'linear':
			trend_gen = LinearTrendGenerator(seed=seed, **trend_params)
		elif trend_type == 'exponential':
			trend_gen = ExponentialTrendGenerator(seed=seed, **trend_params)
		elif trend_type == 'polynomial':
			trend_gen = PolynomialTrendGenerator(seed=seed, **trend_params)
		else:
			raise ValueError(f"Unknown trend_type: {trend_type}")
		
		# Create seasonal generator
		if len(seasonal_periods) == 1:
			seasonal_gen = SeasonalGenerator(period=seasonal_periods[0], amplitude=seasonal_amplitudes[0], noise_std=0,
					# No noise in seasonal component
					seed=seed)
		else:
			seasonal_gen = MultiSeasonalGenerator(periods=seasonal_periods, amplitudes=seasonal_amplitudes, noise_std=0,
					# No noise in seasonal component
					seed=seed)
		
		# Create noise generator
		noise_gen = WhiteNoiseGenerator(mean=0, std=noise_std, seed=seed)
		
		# Initialize composite generator
		super().__init__(generators=[trend_gen, seasonal_gen, noise_gen], combination_method='additive', seed=seed)


class EconomicSeriesGenerator(CompositeGenerator):
	"""
	Generate realistic economic/financial time series.

	Combines multiple components common in economic data:
	- Long-term growth trend
	- Business cycle (multi-year oscillation)
	- Seasonal effects
	- Random shocks and noise

	Parameters
	----------
	initial_level : float, default 100.0
		Starting level of the series
	growth_rate : float, default 0.02
		Annual growth rate (as decimal, e.g., 0.02 = 2%)
	business_cycle_period : int, default 1460
		Business cycle period in time units (default ~4 years daily)
	business_cycle_amplitude : float, default 0.1
		Amplitude of business cycle fluctuations
	seasonal_strength : float, default 0.05
		Strength of seasonal effects
	volatility : float, default 0.02
		Day-to-day volatility
	shock_probability : float, default 0.01
		Probability of economic shocks (1% = rare events)
	shock_magnitude : float, default 0.1
		Magnitude of economic shocks
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # GDP-like series
	>>> gdp_gen = EconomicSeriesGenerator(
	...     initial_level=20000,  # $20T initial GDP
	...     growth_rate=0.025,    # 2.5% annual growth
	...     business_cycle_period=1460,  # 4-year cycles
	...     seasonal_strength=0.02,      # Mild seasonality
	...     volatility=0.01              # Low volatility
	... )
	>>> gdp_data = gdp_gen.generate(n_points=3650)  # 10 years daily
	>>>
	>>> # Stock market index-like series
	>>> market_gen = EconomicSeriesGenerator(
	...     initial_level=4000,
	...     growth_rate=0.08,     # 8% annual growth
	...     volatility=0.015,     # Higher volatility
	...     shock_probability=0.005,  # Market crashes
	...     shock_magnitude=0.2   # 20% crash magnitude
	... )
	>>> market_data = market_gen.generate(n_points=2520)  # ~7 years
	"""
	
	def __init__(self, initial_level: float = 100.0, growth_rate: float = 0.02, business_cycle_period: int = 1460,
	             business_cycle_amplitude: float = 0.1, seasonal_strength: float = 0.05, volatility: float = 0.02,
	             shock_probability: float = 0.01, shock_magnitude: float = 0.1, seed: Optional[int] = None):
		# Convert annual growth rate to daily
		daily_growth_rate = growth_rate / 365.25
		
		# Create components
		trend_gen = ExponentialTrendGenerator(initial_value=initial_level, growth_rate=daily_growth_rate, noise_std=0,
				seed=seed)
		
		# Business cycle (long-term oscillation)
		business_cycle_gen = SeasonalGenerator(period=business_cycle_period,
				amplitude=business_cycle_amplitude * initial_level, noise_std=0, seed=seed)
		
		# Seasonal effects (annual)
		seasonal_gen = SeasonalGenerator(period=365, amplitude=seasonal_strength * initial_level, noise_std=0,
				seed=seed)
		
		# Daily volatility
		volatility_gen = WhiteNoiseGenerator(mean=0, std=volatility * initial_level, seed=seed)
		
		# Economic shocks
		from .noise import OutlierNoiseGenerator
		shock_gen = OutlierNoiseGenerator(base_std=0,  # No base noise
				outlier_probability=shock_probability, outlier_scale=shock_magnitude * initial_level,
				outlier_type='both', seed=seed)
		
		# Combine all components
		super().__init__(generators=[trend_gen, business_cycle_gen, seasonal_gen, volatility_gen, shock_gen],
				combination_method='additive', seed=seed)


class WeatherPatternGenerator(CompositeGenerator):
	"""
	Generate realistic weather-like time series.

	Combines components typical of weather data:
	- Seasonal temperature cycle
	- Daily temperature variation
	- Random weather fluctuations
	- Occasional extreme weather events

	Parameters
	----------
	base_temperature : float, default 15.0
		Average temperature (°C)
	annual_amplitude : float, default 20.0
		Seasonal temperature variation (°C)
	daily_amplitude : float, default 8.0
		Daily temperature variation (°C)
	weather_noise : float, default 2.0
		Random weather fluctuations (°C)
	extreme_probability : float, default 0.02
		Probability of extreme weather (2% chance)
	extreme_magnitude : float, default 15.0
		Magnitude of extreme weather events (°C)
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Temperate climate temperature
	>>> temp_gen = WeatherPatternGenerator(
	...     base_temperature=12.0,  # 12°C average
	...     annual_amplitude=25.0,  # -13°C to +37°C range
	...     daily_amplitude=8.0,    # 8°C daily variation
	...     weather_noise=3.0,      # 3°C random fluctuation
	...     extreme_probability=0.01  # 1% extreme events
	... )
	>>> temp_data = temp_gen.generate(n_points=8760, freq='H')  # 1 year hourly
	>>>
	>>> # Tropical climate
	>>> tropical_gen = WeatherPatternGenerator(
	...     base_temperature=28.0,
	...     annual_amplitude=5.0,   # Less seasonal variation
	...     daily_amplitude=6.0,
	...     extreme_probability=0.005  # Fewer extremes
	... )
	>>> tropical_data = tropical_gen.generate(n_points=365)  # 1 year daily
	"""
	
	def __init__(self, base_temperature: float = 15.0, annual_amplitude: float = 20.0, daily_amplitude: float = 8.0,
	             weather_noise: float = 2.0, extreme_probability: float = 0.02, extreme_magnitude: float = 15.0,
	             seed: Optional[int] = None):
		# Base level (average temperature)
		from .trend import LinearTrendGenerator
		base_gen = LinearTrendGenerator(slope=0,  # No trend
				intercept=base_temperature, noise_std=0, seed=seed)
		
		# Annual seasonal cycle
		annual_gen = SeasonalGenerator(period=365, amplitude=annual_amplitude, phase=np.pi,  # Winter minimum
				noise_std=0, seed=seed)
		
		# Daily temperature cycle
		daily_gen = SeasonalGenerator(period=24, amplitude=daily_amplitude, phase=np.pi / 4,  # Early morning minimum
				noise_std=0, seed=seed)
		
		# Random weather fluctuations
		weather_gen = WhiteNoiseGenerator(mean=0, std=weather_noise, seed=seed)
		
		# Extreme weather events
		from .noise import OutlierNoiseGenerator
		extreme_gen = OutlierNoiseGenerator(base_std=0, outlier_probability=extreme_probability,
				outlier_scale=extreme_magnitude, outlier_type='both', seed=seed)
		
		# Combine all components
		super().__init__(generators=[base_gen, annual_gen, daily_gen, weather_gen, extreme_gen],
				combination_method='additive', seed=seed)


class SalesDataGenerator(CompositeGenerator):
	"""
	Generate realistic sales/business time series.

	Combines components typical of business data:
	- Long-term growth trend
	- Seasonal sales patterns
	- Day-of-week effects
	- Marketing campaign effects
	- Random business fluctuations

	Parameters
	----------
	initial_sales : float, default 1000.0
		Initial sales level
	growth_rate : float, default 0.05
		Annual growth rate (5% = 0.05)
	seasonal_strength : float, default 0.3
		Strength of seasonal effects (30% = 0.3)
	weekly_pattern : bool, default True
		Include weekly sales patterns
	campaign_frequency : float, default 0.1
		Frequency of marketing campaigns (10% = 0.1)
	campaign_boost : float, default 0.5
		Sales boost from campaigns (50% = 0.5)
	base_volatility : float, default 0.1
		Day-to-day sales volatility (10% = 0.1)
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Retail sales with strong seasonality
	>>> retail_gen = SalesDataGenerator(
	...     initial_sales=50000,
	...     growth_rate=0.08,
	...     seasonal_strength=0.4,  # Strong holiday effects
	...     weekly_pattern=True,
	...     campaign_frequency=0.15,  # Frequent campaigns
	...     base_volatility=0.15
	... )
	>>> retail_data = retail_gen.generate(n_points=730)  # 2 years
	>>>
	>>> # B2B sales (less seasonal)
	>>> b2b_gen = SalesDataGenerator(
	...     initial_sales=100000,
	...     seasonal_strength=0.1,  # Weak seasonality
	...     weekly_pattern=False,   # No weekend effects
	...     campaign_frequency=0.05,  # Rare campaigns
	...     base_volatility=0.08    # Lower volatility
	... )
	>>> b2b_data = b2b_gen.generate(n_points=365)
	"""
	
	def __init__(self, initial_sales: float = 1000.0, growth_rate: float = 0.05, seasonal_strength: float = 0.3,
	             weekly_pattern: bool = True, campaign_frequency: float = 0.1, campaign_boost: float = 0.5,
	             base_volatility: float = 0.1, seed: Optional[int] = None):
		
		# Growth trend
		daily_growth = growth_rate / 365.25
		trend_gen = ExponentialTrendGenerator(initial_value=initial_sales, growth_rate=daily_growth, noise_std=0,
				seed=seed)
		
		generators = [trend_gen]
		
		# Seasonal patterns (holiday shopping, etc.)
		seasonal_gen = SeasonalGenerator(period=365, amplitude=seasonal_strength * initial_sales, phase=5.5,
				# Peak in December
				noise_std=0, seed=seed)
		generators.append(seasonal_gen)
		
		# Weekly patterns (if enabled)
		if weekly_pattern:
			weekly_gen = SeasonalGenerator(period=7, amplitude=0.1 * initial_sales, phase=0,  # Peak early in week
					noise_std=0, seed=seed)
			generators.append(weekly_gen)
		
		# Marketing campaigns
		from .noise import OutlierNoiseGenerator
		campaign_gen = OutlierNoiseGenerator(base_std=0, outlier_probability=campaign_frequency,
				outlier_scale=campaign_boost * initial_sales, outlier_type='positive',  # Only positive boosts
				seed=seed)
		generators.append(campaign_gen)
		
		# Base volatility
		volatility_gen = WhiteNoiseGenerator(mean=0, std=base_volatility * initial_sales, seed=seed)
		generators.append(volatility_gen)
		
		# Initialize composite generator
		super().__init__(generators=generators, combination_method='additive', seed=seed)


class CustomCompositeGenerator(CompositeGenerator):
	"""
	Flexible composite generator with custom combination logic.

	Allows users to define custom ways of combining multiple generators,
	including complex mathematical operations and conditional logic.

	Parameters
	----------
	generators : list of TimeSeriesGenerator
		List of generators to combine
	combination_function : callable
		Custom function to combine generator outputs
		Should accept list of pd.Series and return pd.Series
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Custom combination: weighted geometric mean
	>>> def geometric_mean(series_list):
	...     result = series_list[0].copy()
	...     for series in series_list[1:]:
	...         result = result * series.values
	...     return result ** (1.0 / len(series_list))
	>>>
	>>> from datasets.generators.trend import LinearTrendGenerator
	>>> from datasets.generators.seasonal import SeasonalGenerator
	>>>
	>>> trend = LinearTrendGenerator(slope=0.1, intercept=10)
	>>> seasonal = SeasonalGenerator(amplitude=2.0)
	>>>
	>>> custom_gen = CustomCompositeGenerator(
	...     generators=[trend, seasonal],
	...     combination_function=geometric_mean
	... )
	>>> data = custom_gen.generate(n_points=365)
	>>>
	>>> # Complex conditional combination
	>>> def conditional_combine(series_list):
	...     trend, seasonal, noise = series_list
	...     # Use seasonal only when trend is positive
	...     mask = trend.values > 0
	...     result = trend.copy()
	...     result.values[mask] += seasonal.values[mask]
	...     result += noise.values
	...     return result
	"""
	
	def __init__(self, generators: List[TimeSeriesGenerator],
	             combination_function: Callable[[List[pd.Series]], pd.Series], seed: Optional[int] = None):
		super().__init__(generators=generators, combination_method=combination_function, seed=seed)