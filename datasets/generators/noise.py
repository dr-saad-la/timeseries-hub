"""
Time series noise generators.

This module provides generators for various types of noise patterns including
white noise, colored noise, autoregressive noise, heteroscedastic noise, and more.

**Author**: Dr. Saad Laouadi
**Email**: dr.saad.laouadi@gmail.com
**Project**: timeseries-hub
**License**: MIT
"""

from datetime import datetime
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseNoiseGenerator


class WhiteNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate white noise (Gaussian) time series.

	Creates independent, identically distributed random noise from a normal distribution.
	This is the most basic type of noise with no temporal correlation.

	Parameters
	----------
	mean : float, default 0.0
		Mean of the noise distribution
	std : float, default 1.0
		Standard deviation of the noise distribution
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Basic white noise
	>>> gen = WhiteNoiseGenerator(mean=0, std=1.0)
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # Offset white noise (like measurement error)
	>>> gen = WhiteNoiseGenerator(mean=0.5, std=0.2)
	>>> data = gen.generate(n_points=500)
	"""
	
	def __init__(self, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None):
		super().__init__(noise_std=std, seed=seed)
		self.mean = mean
		self.std = std
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate white noise time series.

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
			Time series with white noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		noise = self.rng.normal(self.mean, self.std, n_points)
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(noise, index=index, name='white_noise')


class ColoredNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate colored noise with specific spectral characteristics.

	Creates noise with different frequency characteristics:
	- White noise: β = 0 (flat spectrum)
	- Pink noise: β = 1 (1/f spectrum)
	- Brown/Red noise: β = 2 (1/f² spectrum)

	Parameters
	----------
	beta : float, default 1.0
		Spectral exponent (0=white, 1=pink, 2=brown/red)
	scale : float, default 1.0
		Scaling factor for the noise amplitude
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Pink noise (1/f noise) - common in financial data
	>>> gen = ColoredNoiseGenerator(beta=1.0, scale=1.0)
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # Brown noise (1/f² noise) - like random walk
	>>> gen = ColoredNoiseGenerator(beta=2.0, scale=0.5)
	>>> data = gen.generate(n_points=2000)
	"""
	
	def __init__(self, beta: float = 1.0, scale: float = 1.0, seed: Optional[int] = None):
		super().__init__(noise_std=scale, seed=seed)
		self.beta = beta
		self.scale = scale
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate colored noise time series.

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
			Time series with colored noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate white noise
		white_noise = self.rng.normal(0, 1, n_points)
		
		# Apply spectral shaping using FFT
		if self.beta == 0:
			# White noise - no filtering needed
			colored_noise = white_noise
		else:
			# Create frequency domain filter
			freqs = np.fft.fftfreq(n_points)[1:]  # Exclude DC component
			freqs[0] = 1e-10  # Avoid division by zero
			
			# Apply 1/f^β filter
			filter_response = np.zeros(n_points, dtype=complex)
			filter_response[1:] = freqs ** (-self.beta / 2.0)
			filter_response[0] = 0  # DC component
			
			# Apply filter in frequency domain
			white_fft = np.fft.fft(white_noise)
			colored_fft = white_fft * filter_response
			colored_noise = np.fft.ifft(colored_fft).real
		
		# Scale the noise
		colored_noise = self.scale * colored_noise
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(colored_noise, index=index, name=f'colored_noise_beta_{self.beta}')


class AutoregressiveNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate autoregressive AR(p) noise.

	Creates noise where each value depends on previous values plus white noise.
	AR(1): X(t) = φ₁*X(t-1) + ε(t)
	AR(p): X(t) = φ₁*X(t-1) + ... + φₚ*X(t-p) + ε(t)

	Parameters
	----------
	ar_coeffs : list of float
		Autoregressive coefficients [φ₁, φ₂, ..., φₚ]
	innovation_std : float, default 1.0
		Standard deviation of the innovation (white noise) process
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # AR(1) process with moderate persistence
	>>> gen = AutoregressiveNoiseGenerator(ar_coeffs=[0.7], innovation_std=1.0)
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # AR(2) process
	>>> gen = AutoregressiveNoiseGenerator(
	...     ar_coeffs=[0.5, 0.3],
	...     innovation_std=0.5
	... )
	>>> data = gen.generate(n_points=2000)
	"""
	
	def __init__(self, ar_coeffs: List[float], innovation_std: float = 1.0, seed: Optional[int] = None):
		super().__init__(noise_std=innovation_std, seed=seed)
		self.ar_coeffs = list(np.array(ar_coeffs))
		self.innovation_std = innovation_std
		self.order = len(ar_coeffs)
		
		# Check stability condition
		if not self._check_stability():
			import warnings
			warnings.warn("AR coefficients may lead to unstable process", UserWarning)
	
	def _check_stability(self) -> bool:
		"""Check if AR process is stationary (simplified check)."""
		if self.order == 1:
			return abs(self.ar_coeffs[0]) < 1
		# For higher order, this is a simplified check
		return np.sum(np.abs(self.ar_coeffs)) < 1
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate autoregressive noise time series.

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
			Time series with autoregressive noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Initialize the series
		series = np.zeros(n_points)
		innovations = self.rng.normal(0, self.innovation_std, n_points)
		
		# Generate AR process
		for t in range(n_points):
			# Add innovation
			series[t] = innovations[t]
			
			# Add autoregressive terms
			for i, coeff in enumerate(self.ar_coeffs):
				if t - i - 1 >= 0:
					series[t] += coeff * series[t - i - 1]
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(series, index=index, name=f'ar_{self.order}_noise')


class MovingAverageNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate moving average MA(q) noise.

	Creates noise as a weighted sum of current and past white noise innovations.
	MA(q): X(t) = ε(t) + θ₁*ε(t-1) + ... + θᵣ*ε(t-q)

	Parameters
	----------
	ma_coeffs : list of float
		Moving average coefficients [θ₁, θ₂, ..., θᵣ]
	innovation_std : float, default 1.0
		Standard deviation of the innovation (white noise) process
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # MA(1) process
	>>> gen = MovingAverageNoiseGenerator(ma_coeffs=[0.5], innovation_std=1.0)
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # MA(3) process with different weights
	>>> gen = MovingAverageNoiseGenerator(
	...     ma_coeffs=[0.6, 0.3, 0.1],
	...     innovation_std=0.8
	... )
	>>> data = gen.generate(n_points=1500)
	"""
	
	def __init__(self, ma_coeffs: List[float], innovation_std: float = 1.0, seed: Optional[int] = None):
		super().__init__(noise_std=innovation_std, seed=seed)
		self.ma_coeffs = list(np.array(ma_coeffs))
		self.innovation_std = innovation_std
		self.order = len(ma_coeffs)
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate moving average noise time series.

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
			Time series with moving average noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate innovations
		innovations = self.rng.normal(0, self.innovation_std, n_points)
		
		# Initialize the series
		series = np.zeros(n_points)
		
		# Generate MA process
		for t in range(n_points):
			# Current innovation
			series[t] = innovations[t]
			
			# Add moving average terms
			for i, coeff in enumerate(self.ma_coeffs):
				if t - i - 1 >= 0:
					series[t] += coeff * innovations[t - i - 1]
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(series, index=index, name=f'ma_{self.order}_noise')


class HeteroscedasticNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate heteroscedastic noise with time-varying variance.

	Creates noise where the variance changes over time, common in financial data.
	Can model GARCH-like effects or seasonal variance changes.

	Parameters
	----------
	variance_function : callable or str
		Function defining variance over time, or predefined pattern name.
		If callable: func(t) -> variance
		If str: 'linear', 'seasonal', 'exponential'
	base_variance : float, default 1.0
		Base variance level
	variance_scale : float, default 0.5
		Scaling factor for variance changes
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Linearly increasing variance
	>>> gen = HeteroscedasticNoiseGenerator(
	...     variance_function='linear',
	...     base_variance=1.0,
	...     variance_scale=0.002
	... )
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # Custom variance function
	>>> def custom_var(t):
	...     return 1 + 0.5 * np.sin(2 * np.pi * t / 50)
	>>>
	>>> gen = HeteroscedasticNoiseGenerator(variance_function=custom_var)
	>>> data = gen.generate(n_points=500)
	"""
	
	def __init__(self, variance_function: Union[Callable, str] = 'linear', base_variance: float = 1.0,
	             variance_scale: float = 0.5, seed: Optional[int] = None):
		super().__init__(noise_std=np.sqrt(base_variance), seed=seed)
		self.variance_function = variance_function
		self.base_variance = base_variance
		self.variance_scale = variance_scale
	
	def _get_variance(self, t: np.ndarray) -> np.ndarray:
		"""Calculate variance for each time point."""
		if callable(self.variance_function):
			return self.variance_function(t)
		elif self.variance_function == 'linear':
			return self.base_variance + self.variance_scale * t
		elif self.variance_function == 'seasonal':
			return self.base_variance + self.variance_scale * np.sin(2 * np.pi * t / 365)
		elif self.variance_function == 'exponential':
			return self.base_variance * np.exp(self.variance_scale * t / len(t))
		else:
			raise ValueError(f"Unknown variance function: {self.variance_function}")
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate heteroscedastic noise time series.

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
			Time series with heteroscedastic noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		t = np.arange(n_points)
		variances = self._get_variance(t)
		
		# Ensure positive variances
		variances = np.maximum(variances, 0.01)
		
		# Generate noise with time-varying variance
		noise = np.zeros(n_points)
		for i in range(n_points):
			noise[i] = self.rng.normal(0, np.sqrt(variances[i]))
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(noise, index=index, name='heteroscedastic_noise')


class OutlierNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate noise with occasional outliers.

	Creates mostly normal noise with occasional large spikes/outliers,
	simulating real-world data with anomalies or extreme events.

	Parameters
	----------
	base_std : float, default 1.0
		Standard deviation of the base noise
	outlier_probability : float, default 0.05
		Probability of outlier occurrence (0.05 = 5% chance)
	outlier_scale : float, default 5.0
		Multiplier for outlier magnitude (relative to base_std)
	outlier_type : str, default 'both'
		Type of outliers: 'positive', 'negative', or 'both'
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # 5% chance of outliers, 5x normal magnitude
	>>> gen = OutlierNoiseGenerator(
	...     base_std=1.0,
	...     outlier_probability=0.05,
	...     outlier_scale=5.0
	... )
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # Only positive outliers (spikes up)
	>>> gen = OutlierNoiseGenerator(
	...     outlier_type='positive',
	...     outlier_probability=0.02
	... )
	>>> data = gen.generate(n_points=2000)
	"""
	
	def __init__(self, base_std: float = 1.0, outlier_probability: float = 0.05, outlier_scale: float = 5.0,
	             outlier_type: str = 'both', seed: Optional[int] = None):
		super().__init__(noise_std=base_std, seed=seed)
		self.base_std = base_std
		self.outlier_probability = outlier_probability
		self.outlier_scale = outlier_scale
		self.outlier_type = outlier_type
		
		if outlier_type not in ['positive', 'negative', 'both']:
			raise ValueError("outlier_type must be 'positive', 'negative', or 'both'")
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate noise with outliers.

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
			Time series with outlier noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate base noise
		noise = self.rng.normal(0, self.base_std, n_points)
		
		# Generate outlier mask
		outlier_mask = self.rng.random(n_points) < self.outlier_probability
		
		# Add outliers
		for i in range(n_points):
			if outlier_mask[i]:
				if self.outlier_type == 'positive':
					outlier_magnitude = self.outlier_scale * self.base_std
				elif self.outlier_type == 'negative':
					outlier_magnitude = -self.outlier_scale * self.base_std
				else:  # 'both'
					sign = 1 if self.rng.random() > 0.5 else -1
					outlier_magnitude = sign * self.outlier_scale * self.base_std
				
				noise[i] = outlier_magnitude
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(noise, index=index, name='outlier_noise')


class StudentTNoiseGenerator(BaseNoiseGenerator):
	"""
	Generate Student's t-distributed noise.

	Creates noise from Student's t-distribution, which has heavier tails
	than normal distribution, making it more realistic for financial data.

	Parameters
	----------
	degrees_freedom : float, default 5.0
		Degrees of freedom (lower = heavier tails, higher = closer to normal)
	scale : float, default 1.0
		Scale parameter of the distribution
	seed : int, optional
		Random seed for reproducible generation

	Examples
	--------
	>>> # Heavy-tailed noise (more extreme values)
	>>> gen = StudentTNoiseGenerator(degrees_freedom=3.0, scale=1.0)
	>>> data = gen.generate(n_points=1000)
	>>>
	>>> # Closer to normal distribution
	>>> gen = StudentTNoiseGenerator(degrees_freedom=30.0, scale=0.5)
	>>> data = gen.generate(n_points=1500)
	"""
	
	def __init__(self, degrees_freedom: float = 5.0, scale: float = 1.0, seed: Optional[int] = None):
		super().__init__(noise_std=scale, seed=seed)
		self.degrees_freedom = degrees_freedom
		self.scale = scale
	
	def generate(self, n_points: int, start_date: Optional[str] = None, freq: str = 'D', **kwargs) -> pd.Series:
		"""
		Generate Student's t-distributed noise.

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
			Time series with Student's t noise
		"""
		if start_date is None:
			start_date = f"{datetime.now().year}-01-01"
		
		# Generate t-distributed noise
		noise = self.rng.standard_t(self.degrees_freedom, n_points) * self.scale
		
		index = self._create_datetime_index(n_points, start_date, freq)
		return pd.Series(noise, index=index, name='student_t_noise')
