"""
Base classes for time series data generators.

This module provides abstract base classes and common functionality for generating
synthetic time series data with various patterns and characteristics.

**Author**: Dr. Saad Laouadi
**Email**: dr.saad.laouadi@gmail.com
**Project**: timeseries-hub
**License**: MIT
"""
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd


class TimeSeriesGenerator(ABC):
	"""
	Abstract base class for all time series generators.

	This class provides a common interface for generating synthetic time series
	data with different patterns, trends, and noise characteristics.

	Parameters
	----------
	seed : int, optional
		Random seed for reproducible generation, by default None

	Attributes
	----------
	seed : int or None
		Random seed used for generation
	rng : numpy.random.RandomState
		Random number generator instance
	params : dict
		Dictionary storing generator parameters
	"""
	
	def __init__(self, seed: Optional[int] = None):
		self.seed = seed
		self.rng = np.random.RandomState(seed)
		self.params = {}
	
	@abstractmethod
	def generate(self, n_points: int, **kwargs) -> pd.Series:
		"""
		Generate synthetic time series data.

		Parameters
		----------
		n_points : int
			Number of data points to generate
		**kwargs
			Additional keyword arguments specific to each generator

		Returns
		-------
		pd.Series
			Generated time series with datetime index
		"""
		pass
	
	def set_params(self, **params) -> "TimeSeriesGenerator":
		"""
		Set parameters for the generator.

		Parameters
		----------
		**params
			Parameter names and values to set

		Returns
		-------
		TimeSeriesGenerator
			Self for method chaining
		"""
		self.params.update(params)
		return self
	
	def get_params(self) -> Dict[str, Any]:
		"""
		Get current parameters.

		Returns
		-------
		dict
			Copy of current parameters
		"""
		return self.params.copy()
	
	def set_seed(self, seed: int) -> "TimeSeriesGenerator":
		"""
		Set random seed for reproducible generation.

		Parameters
		----------
		seed : int
			Random seed value

		Returns
		-------
		TimeSeriesGenerator
			Self for method chaining
		"""
		self.seed = seed
		self.rng = np.random.RandomState(seed)
		return self
	
	@staticmethod
	def _create_datetime_index(n_points: int, start_date: Optional[str] = None,
	                           freq: str = "D") -> pd.DatetimeIndex:
		"""
		Create datetime index for time series.

		Parameters
		----------
		n_points : int
			Number of points in the series
		start_date : str, optional
            Starting date for the series. If None, uses current year.
		freq : str, default 'D'
			Frequency string (D=daily, H=hourly, etc.)

		Returns
		-------
		pd.DatetimeIndex
			Datetime index for the time series
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		return pd.date_range(start_date, periods=n_points, freq=freq)


class BaseNoiseGenerator(TimeSeriesGenerator, ABC):
	"""
	Base class for noise generators.

	Provides common functionality for adding various types of noise
	to time series data.
	"""
	
	def __init__(self, noise_std: float = 0.1, seed: Optional[int] = None):
		super().__init__(seed)
		self.noise_std = noise_std
	
	def add_noise(self, signal: np.ndarray) -> np.ndarray:
		"""
		Add noise to a signal.

		Parameters
		----------
		signal : np.ndarray
			Input signal

		Returns
		-------
		np.ndarray
			Signal with added noise
		"""
		noise = self.rng.normal(0, self.noise_std, len(signal))
		return signal + noise
