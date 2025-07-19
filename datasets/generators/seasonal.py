"""
Time series seasonal pattern generators.

This module provides generators for various types of seasonal patterns including
simple sinusoidal, multi-seasonal, additive/multiplicative, and custom patterns.

**Author**: Dr. Saad Laouadi
**Email**: dr.saad.laouadi@gmail.com
**Project**: timeseries-hub
**License**: MIT
"""

from datetime import datetime
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from .base import TimeSeriesGenerator


class SeasonalGenerator(TimeSeriesGenerator):
	"""
	Generate time series with simple seasonal patterns.

	Creates sinusoidal seasonal patterns with configurable period, amplitude, and phase.

	Parameters
	----------
	period : int, default 12
		Number of time steps for one complete cycle
	amplitude : float, default 1.0
		Amplitude of the seasonal pattern
	phase : float, default 0.0
		Phase shift in radians (0 = starts at 0, π/2 = starts at peak)
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Monthly seasonal pattern (12-month cycle)
	>>> gen = SeasonalGenerator(period=12, amplitude=2.0, phase=0)
	>>> data = gen.generate(n_points=365)
	>>>
	>>> # Daily pattern (24-hour cycle)
	>>> gen = SeasonalGenerator(period=24, amplitude=5.0)
	>>> data = gen.generate(n_points=168, freq='H')  # One week hourly
	"""
	
	def __init__(self, period: int = 12, amplitude: float = 1.0, phase: float = 0.0, noise_std: float = 0.1,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.period = period
		self.amplitude = amplitude
		self.phase = phase
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate seasonal time series.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string (D=daily, H=hourly, etc.)
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series
			Time series with seasonal pattern
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		seasonal = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(seasonal + noise, index=index, name='seasonal')


class MultiSeasonalGenerator(TimeSeriesGenerator):
	"""
	Generate time series with multiple overlapping seasonal patterns.

	Combines multiple seasonal components with different periods and amplitudes.

	Parameters
	----------
	periods : list of int
		List of periods for each seasonal component
	amplitudes : list of float
		List of amplitudes for each seasonal component
	phases : list of float, optional
		List of phase shifts for each component. If None, all phases are 0.
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Weekly + yearly seasonality
	>>> gen = MultiSeasonalGenerator(
	...     periods=[7, 365],
	...     amplitudes=[1.0, 3.0],
	...     phases=[0, np.pi/4]
	... )
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # Hourly + daily + weekly patterns
	>>> gen = MultiSeasonalGenerator(
	...     periods=[24, 168, 8760],  # 1 day, 1 week, 1 year (in hours)
	...     amplitudes=[2.0, 1.5, 0.8]
	... )
	>>> data = gen.generate(n_points=8760, freq='H')  # One year hourly
	"""
	
	def __init__(self, periods: List[int], amplitudes: List[float], phases: Optional[List[float]] = None,
	             noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		
		if len(periods) != len(amplitudes):
			raise ValueError("periods and amplitudes must have the same length")
		
		self.periods = periods
		self.amplitudes = amplitudes
		self.phases = phases if phases is not None else [0.0] * len(periods)
		self.noise_std = noise_std
		
		if len(self.phases) != len(periods):
			raise ValueError("phases must have the same length as periods")
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate multi-seasonal time series.

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
			Time series with multiple seasonal patterns
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		signal = np.zeros(n_points)
		
		# Sum all seasonal components
		for period, amplitude, phase in zip(self.periods, self.amplitudes, self.phases):
			signal += amplitude * np.sin(2 * np.pi * t / period + phase)
		
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(signal + noise, index=index, name='multi_seasonal')


class AdditiveSeasonalGenerator(TimeSeriesGenerator):
	"""
	Generate time series with additive seasonal effects.

	Creates patterns where seasonal effects are added to a base level.
	Good for modeling patterns where seasonal variation is constant.

	Parameters
	----------
	base_level : float, default 0.0
		Base level around which seasonal variation occurs
	seasonal_amplitudes : list of float, optional
		Amplitudes for each seasonal component. If None, uses [1.0].
	seasonal_periods : list of int, optional
		Periods for each seasonal component. If None, uses [12].
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Temperature-like pattern with base + seasonal variation
	>>> gen = AdditiveSeasonalGenerator(
	...     base_level=20.0,  # 20°C average
	...     seasonal_amplitudes=[15.0, 2.0],  # Yearly and daily variation
	...     seasonal_periods=[365, 1]  # 365 days and daily cycle
	... )
	>>> data = gen.generate(n_points=365)
	"""
	
	def __init__(self, base_level: float = 0.0, seasonal_amplitudes: Optional[List[float]] = None,
	             seasonal_periods: Optional[List[int]] = None, noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		
		# Use safe defaults
		if seasonal_amplitudes is None:
			seasonal_amplitudes = [1.0]
		if seasonal_periods is None:
			seasonal_periods = [12]
		
		if len(seasonal_amplitudes) != len(seasonal_periods):
			raise ValueError("seasonal_amplitudes and seasonal_periods must have same length")
		
		self.base_level = base_level
		self.seasonal_amplitudes = seasonal_amplitudes
		self.seasonal_periods = seasonal_periods
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate additive seasonal time series.

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
			Time series with additive seasonal effects
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		
		# Start with base level
		signal = np.full(n_points, self.base_level)
		
		# Add seasonal components
		for amplitude, period in zip(self.seasonal_amplitudes, self.seasonal_periods):
			signal += amplitude * np.sin(2 * np.pi * t / period)
		
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(signal + noise, index=index, name='additive_seasonal')


class MultiplicativeSeasonalGenerator(TimeSeriesGenerator):
	"""
	Generate time series with multiplicative seasonal effects.

	Creates patterns where seasonal effects multiply the base level.
	Good for modeling patterns where seasonal variation scales with the level.

	Parameters
	----------
	base_level : float, default 1.0
		Base level that gets multiplied by seasonal factors
	seasonal_factors : list of float, optional
		Multiplicative factors for each seasonal component (1.0 = no effect).
		If None, uses [0.2].
	seasonal_periods : list of int, optional
		Periods for each seasonal component. If None, uses [12].
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise (applied multiplicatively)
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Sales pattern that scales with base level
	>>> gen = MultiplicativeSeasonalGenerator(
	...     base_level=1000.0,  # Base sales level
	...     seasonal_factors=[0.3, 0.1],  # 30% yearly, 10% weekly variation
	...     seasonal_periods=[52, 7]  # 52 weeks, 7 days
	... )
	>>> data = gen.generate(n_points=365)
	"""
	
	def __init__(self, base_level: float = 1.0, seasonal_factors: Optional[List[float]] = None,
	             seasonal_periods: Optional[List[int]] = None, noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		
		# Use safe defaults
		if seasonal_factors is None:
			seasonal_factors = [0.2]
		if seasonal_periods is None:
			seasonal_periods = [12]
		
		if len(seasonal_factors) != len(seasonal_periods):
			raise ValueError("seasonal_factors and seasonal_periods must have same length")
		
		self.base_level = base_level
		self.seasonal_factors = seasonal_factors
		self.seasonal_periods = seasonal_periods
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate multiplicative seasonal time series.

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
			Time series with multiplicative seasonal effects
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		
		# Start with base level
		signal = np.full(n_points, self.base_level)
		
		# Apply multiplicative seasonal effects
		for factor, period in zip(self.seasonal_factors, self.seasonal_periods):
			seasonal_multiplier = 1 + factor * np.sin(2 * np.pi * t / period)
			signal *= seasonal_multiplier
		
		# Apply multiplicative noise
		noise_multiplier = 1 + self.rng.normal(0, self.noise_std, n_points)
		signal *= noise_multiplier
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(signal, index=index, name='multiplicative_seasonal')


class IrregularSeasonalGenerator(TimeSeriesGenerator):
	"""
	Generate time series with irregular seasonal patterns.

	Creates seasonal patterns with random variations in amplitude and timing,
	simulating real-world seasonal patterns that aren't perfectly regular.

	Parameters
	----------
	base_period : int, default 12
		Base period for the seasonal pattern
	amplitude_variation : float, default 0.2
		Random variation in amplitude (0 = no variation, 1 = 100% variation)
	period_variation : float, default 0.1
		Random variation in period (0 = no variation, 0.1 = 10% variation)
	phase_drift : float, default 0.0
		Gradual drift in phase over time
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Irregular seasonal pattern (like real weather)
	>>> gen = IrregularSeasonalGenerator(
	...     base_period=365,
	...     amplitude_variation=0.3,  # 30% variation in seasonal strength
	...     period_variation=0.05,    # 5% variation in timing
	...     phase_drift=0.01
	... )
	>>> data = gen.generate(n_points=1095)  # 3 years
	"""
	
	def __init__(self, base_period: int = 12, amplitude_variation: float = 0.2, period_variation: float = 0.1,
	             phase_drift: float = 0.0, noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		self.base_period = base_period
		self.amplitude_variation = amplitude_variation
		self.period_variation = period_variation
		self.phase_drift = phase_drift
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate irregular seasonal time series.

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
			Time series with irregular seasonal pattern
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		
		# Create varying amplitude
		amplitude_noise = 1 + self.amplitude_variation * self.rng.normal(0, 1, n_points)
		
		# Create varying period
		period_noise = self.base_period * (1 + self.period_variation * self.rng.normal(0, 0.1, n_points))
		
		# Create phase drift
		phase = self.phase_drift * t
		
		# Generate irregular seasonal pattern
		signal = np.zeros(n_points)
		for i in range(n_points):
			signal[i] = amplitude_noise[i] * np.sin(2 * np.pi * t[i] / period_noise[i] + phase[i])
		
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(signal + noise, index=index, name='irregular_seasonal')


class CustomSeasonalGenerator(TimeSeriesGenerator):
	"""
	Generate time series with custom seasonal patterns.

	Allows users to define custom seasonal patterns using functions or lookup tables.

	Parameters
	----------
	seasonal_function : callable
		Function that takes time index and optionally period, returns seasonal values.
		Can have signature: func(t) or func(t, period=None)
		Should accept numpy array and return numpy array of same length
	period : int, default 12
		Period for repeating the custom pattern
	amplitude : float, default 1.0
		Amplitude scaling factor for the custom pattern
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Square wave seasonal pattern (with period parameter)
	>>> def square_wave(t, period=12):
	...     return np.sign(np.sin(2 * np.pi * t / period))
	>>>
	>>> gen = CustomSeasonalGenerator(
	...     seasonal_function=square_wave,
	...     period=12,
	...     amplitude=2.0
	... )
	>>> data = gen.generate(n_points=100)
	>>>
	>>> # Simple function without period parameter
	>>> def simple_pattern(t):
	...     return np.sin(2 * np.pi * t / 24)  # Fixed 24-hour cycle
	>>>
	>>> gen = CustomSeasonalGenerator(seasonal_function=simple_pattern)
	>>> data = gen.generate(n_points=168, freq='H')
	>>>
	>>> # Sawtooth wave pattern
	>>> def sawtooth(t, period=24):
	...     return 2 * (t % period) / period - 1
	>>>
	>>> gen = CustomSeasonalGenerator(seasonal_function=sawtooth, period=24)
	>>> data = gen.generate(n_points=168, freq='H')  # Weekly hourly data
	"""
	
	def __init__(self, seasonal_function: Callable, period: int = 12, amplitude: float = 1.0, noise_std: float = 0.1,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.seasonal_function = seasonal_function
		self.period = period
		self.amplitude = amplitude
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate custom seasonal time series.

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
			Time series with custom seasonal pattern
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		
		# Apply custom seasonal function - try with period first, then without
		try:
			# First try: function accepts period parameter
			import inspect
			sig = inspect.signature(self.seasonal_function)
			if 'period' in sig.parameters:
				seasonal = self.seasonal_function(t, period=self.period)
			else:
				seasonal = self.seasonal_function(t)
			
			seasonal = self.amplitude * np.array(seasonal)
		except Exception as e:
			raise ValueError(f"Error applying seasonal function: {e}")
		
		if len(seasonal) != n_points:
			raise ValueError("Seasonal function must return array of same length as input")
		
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(seasonal + noise, index=index, name='custom_seasonal')
