"""Time series generators package."""

# TODO:
#    Add GARCHGenerator module
#    Maked: Done
#    Other modules:

from .trend import (
    LinearTrendGenerator,
    ExponentialTrendGenerator,
    PolynomialTrendGenerator,
)

from .seasonal import (
    SeasonalGenerator,
    MultiSeasonalGenerator,
    AdditiveSeasonalGenerator,
)

from .noise import WhiteNoiseGenerator, ColoredNoiseGenerator

from .composite import (
    CompositeGenerator,
    TrendSeasonalGenerator,
    EconomicSeriesGenerator,
)

from .financial import (
    GeometricBrownianMotionGenerator,
    MeanRevertingGenerator,
    JumpDiffusionGenerator,
    GARCHGenerator,
)

__all__ = [  # Trend generators
    "LinearTrendGenerator",
    "ExponentialTrendGenerator",
    "PolynomialTrendGenerator",
    # Seasonal generators
    "SeasonalGenerator",
    "MultiSeasonalGenerator",
    "AdditiveSeasonalGenerator",
    # Noise generators
    "WhiteNoiseGenerator",
    "ColoredNoiseGenerator",
    "GARCHGenerator",
    # Composite generators
    "CompositeGenerator",
    "TrendSeasonalGenerator",
    "EconomicSeriesGenerator",
    # Financial generators
    "GeometricBrownianMotionGenerator",
    "MeanRevertingGenerator",
    "JumpDiffusionGenerator",
]
