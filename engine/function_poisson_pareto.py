import pandas as pd
from scipy.stats import pareto, poisson


def get_poisson_pareto_ylt(
    poisson_rate: float, pareto_shape: float, pareto_scale: float, no_year: int
) -> pd.DataFrame:
    """
    Generate a DataFrame with years and corresponding loss amounts based on a Poisson-Pareto distribution.

    This function simulates the number of events per year using a Poisson distribution
    and the loss amounts for these events using a Pareto distribution.

    For more information:
    - Poisson distribution: https://en.wikipedia.org/wiki/Poisson_distribution
    - Pareto distribution: https://en.wikipedia.org/wiki/Pareto_distribution

    :param poisson_rate: The rate parameter (lambda) for the Poisson distribution.
    :param pareto_shape: The shape parameter (alpha) for the Pareto distribution.
    :param pareto_scale: The scale parameter (x_m) for the Pareto distribution.
    :param no_year: The number of years to simulate.
    :return: A Data auitFrame with 'year' and 'loss_amount' columns.
    """
    # Generate frequencies using Poisson distribution
    frequencies = poisson.rvs(poisson_rate, size=no_year)

    # Generate loss amounts using Pareto distribution
    loss_amounts = pareto.rvs(pareto_shape, scale=pareto_scale, size=frequencies.sum())

    # Generate the years corresponding to the frequencies
    years = [
        year for year, frequency in enumerate(frequencies) for _ in range(frequency)
    ]

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "year": years,
            "loss_amount": loss_amounts,
        }
    )

    return df
