"""
Time series trend generators.

This module provides generators for various types of trends including linear,
polynomial, exponential, logarithmic, and piecewise trends.

**Author**: Dr. Saad Laouadi
**Email**: dr.saad.laouadi@gmail.com
**Project**: timeseries-hub
**License**: MIT
"""
from datetime import datetime

from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from .base import TimeSeriesGenerator


class LinearTrendGenerator(TimeSeriesGenerator):
	"""
	Generate time series with linear trend.

	Creates time series following the pattern: y(t) = slope * t + intercept + noise

	Parameters
	----------
	slope : float, default 1.0
		The slope of the linear trend
	intercept : float, default 0.0
		The y-intercept of the linear trend
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> gen = LinearTrendGenerator(slope=0.5, intercept=10, noise_std=0.2)
	>>> data = gen.generate(n_points=100)
	>>> print(data.head())
	"""
	
	def __init__(self, slope: float = 1.0, intercept: float = 0.0,
	             noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		self.slope = slope
		self.intercept = intercept
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None,
	             freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate linear trend time series.

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
			Time series with linear trend
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
			
		t = np.arange(n_points)
		trend = self.slope * t + self.intercept
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(trend + noise, index=index, name='linear_trend')


class PolynomialTrendGenerator(TimeSeriesGenerator):
	"""
	Generate time series with polynomial trend.

	Creates time series following polynomial patterns of any degree.

	Parameters
	----------
	coefficients : list or array-like
		Polynomial coefficients in descending order of powers
		[a_n, a_(n-1), ..., a_1, a_0] for a_n*x^n + ... + a_1*x + a_0
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Quadratic trend: 0.01*t^2 + 0.5*t + 10
	>>> gen = PolynomialTrendGenerator(coefficients=[0.01, 0.5, 10])
	>>> data = gen.generate(n_points=100)
	"""
	
	def __init__(self, coefficients: List[float], noise_std: float = 0.1,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.coefficients = np.array(coefficients)
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] =None ,
	             freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate polynomial trend time series.

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
			Time series with polynomial trend
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
			
		t = np.arange(n_points)
		trend = np.polyval(self.coefficients, t)
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(trend + noise, index=index, name='polynomial_trend')


class ExponentialTrendGenerator(TimeSeriesGenerator):
	"""
	Generate time series with exponential trend.

	Creates time series following: y(t) = initial_value * exp(growth_rate * t) + noise

	Parameters
	----------
	initial_value : float, default 1.0
		Initial value at t=0
	growth_rate : float, default 0.01
		Exponential growth rate (positive for growth, negative for decay)
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Exponential growth
	>>> gen = ExponentialTrendGenerator(initial_value=100, growth_rate=0.02)
	>>> data = gen.generate(n_points=100)
	"""
	
	def __init__(self, initial_value: float = 1.0, growth_rate: float = 0.01, noise_std: float = 0.1,
	             seed: Optional[int] = None):
		super().__init__(seed)
		self.initial_value = initial_value
		self.growth_rate = growth_rate
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None,
	             freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate exponential trend time series.

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
			Time series with exponential trend
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
			
		t = np.arange(n_points)
		trend = self.initial_value * np.exp(self.growth_rate * t)
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(trend + noise, index=index, name='exponential_trend')


class LogarithmicTrendGenerator(TimeSeriesGenerator):
	"""
	Generate time series with logarithmic trend.

	Creates time series following: y(t) = a * log(t + 1) + b + noise

	Parameters
	----------
	scale : float, default 1.0
		Scaling factor for the logarithm
	shift : float, default 0.0
		Vertical shift of the trend
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> gen = LogarithmicTrendGenerator(scale=2.0, shift=5.0)
	>>> data = gen.generate(n_points=100)
	"""
	
	def __init__(self, scale: float = 1.0, shift: float = 0.0,
	             noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		self.scale = scale
		self.shift = shift
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None,
	             freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate logarithmic trend time series.

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
			Time series with logarithmic trend
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		trend = self.scale * np.log(t + 1) + self.shift
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(trend + noise, index=index, name='logarithmic_trend')


class PiecewiseLinearTrendGenerator(TimeSeriesGenerator):
	"""
	Generate time series with piecewise linear trend.

	Creates time series with different linear segments connected at breakpoints.

	Parameters
	----------
	breakpoints : list of int
		Time points where the trend changes
	slopes : list of float
		Slope for each segment (must have len(breakpoints) + 1 elements)
	intercept : float, default 0.0
		Starting value of the first segment
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Three segments with different slopes
	>>> gen = PiecewiseLinearTrendGenerator(
	...     breakpoints=[30, 70],
	...     slopes=[0.5, -0.2, 0.8],
	...     intercept=10
	... )
	>>> data = gen.generate(n_points=100)
	"""
	
	def __init__(self, breakpoints: List[int], slopes: List[float],
	             intercept: float = 0.0, noise_std: float = 0.1,
	             seed: Optional[int] = None):
		super().__init__(seed)
		
		if len(slopes) != len(breakpoints) + 1:
			raise ValueError("Number of slopes must be len(breakpoints) + 1")
		
		self.breakpoints = sorted(breakpoints)
		self.slopes = slopes
		self.intercept = intercept
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] = None,
	             freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate piecewise linear trend time series.

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
			Time series with piecewise linear trend
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
			
		t = np.arange(n_points)
		trend = np.zeros(n_points)
		
		# Calculate trend values
		current_value = self.intercept
		prev_breakpoint = 0
		
		for i, (slope, breakpoint) in enumerate(zip(self.slopes, self.breakpoints + [n_points])):
			segment_length = min(breakpoint, n_points) - prev_breakpoint
			segment_indices = np.arange(prev_breakpoint, min(breakpoint, n_points))
			
			if len(segment_indices) > 0:
				segment_trend = current_value + slope * np.arange(segment_length)
				trend[segment_indices] = segment_trend
				
				if breakpoint < n_points:
					current_value = segment_trend[-1]
			
			prev_breakpoint = breakpoint
		
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(trend + noise, index=index, name='piecewise_linear_trend')


class ChangePointTrendGenerator(TimeSeriesGenerator):
	"""
	Generate time series with trend change points.

	Creates time series where the trend changes abruptly at specific points.

	Parameters
	----------
	change_points : list of int
		Time points where trend changes occur
	trend_values : list of float
		Trend slope for each segment
	initial_level : float, default 0.0
		Starting level of the series
	noise_std : float, default 0.1
		Standard deviation of Gaussian noise
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> gen = ChangePointTrendGenerator(
	...     change_points=[25, 75],
	...     trend_values=[0.1, -0.05, 0.15],
	...     initial_level=50
	... )
	>>> data = gen.generate(n_points=100)
	"""
	
	def __init__(self, change_points: List[int], trend_values: List[float], initial_level: float = 0.0,
	             noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		
		if len(trend_values) != len(change_points) + 1:
			raise ValueError("Number of trend values must be len(change_points) + 1")
		
		self.change_points = sorted(change_points)
		self.trend_values = trend_values
		self.initial_level = initial_level
		self.noise_std = noise_std
	
	def generate(self, n_points: int, start_date: Optional[str] =None,
	             freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate change point trend time series.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		start_date : str, Optional
			Starting date for the time series. If None, uses current year.
		freq : str, default 'D'
			Frequency string
		**kwargs
			Additional keyword arguments

		Returns
		-------
		pd.Series
			Time series with change point trends
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
			
		trend = np.zeros(n_points)
		current_level = self.initial_level
		prev_point = 0
		
		for i, (trend_val, change_point) in enumerate(zip(self.trend_values, self.change_points + [n_points])):
			end_point = min(change_point, n_points)
			segment_length = end_point - prev_point
			
			if segment_length > 0:
				segment_trend = current_level + trend_val * np.arange(segment_length)
				trend[prev_point:end_point] = segment_trend
				
				if end_point < n_points:
					current_level = segment_trend[-1]
			
			prev_point = change_point
		
		noise = self.rng.normal(0, self.noise_std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(trend + noise, index=index, name='changepoint_trend')
