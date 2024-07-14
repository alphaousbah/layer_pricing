import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import LayerYearLoss

PERCENTILES = [
    0.999,
    0.998,
    0.996,
    0.995,
    0.99,
    0.98,
    0.9667,
    0.96,
    0.95,
    0.9,
    0.8,
    0.5,
]


def get_df_resultlayerstatisticloss(
    session: Session, layer_id: int, simulated_years: int
) -> pd.DataFrame:
    """
    Retrieve and calculate the result layer statistic losses for a specified layer.

    This function retrieves the year loss data for the specified layer and calculates
    both the Aggregate Exceedance Probability (AEP) and Occurrence Exceedance Probability (OEP)
    losses. The results are concatenated into a single DataFrame with an added layer ID column.

    :param session: The SQLAlchemy Session for database operations.
    :param layer_id: The ID of the layer to retrieve the statistic loss data for.
    :param simulated_years: The number of simulated years for the analysis.
    :return: A DataFrame containing the concatenated AEP and OEP loss data for the specified layer.
    """
    df_layeryearloss = get_df_layeryearloss(session, layer_id)
    df_aep = get_df_loss(df_layeryearloss, simulated_years, "AEP")
    df_oep = get_df_loss(df_layeryearloss, simulated_years, "OEP")
    # noinspection PyUnreachableCode
    df = pd.concat([df_aep, df_oep])
    df["layer_id"] = layer_id
    return df


def get_df_layeryearloss(session: Session, layer_id: int) -> pd.DataFrame:
    """
    Retrieve the year loss data for a specified layer from the database.

    This function queries the database for year loss data corresponding to the given layer ID.
    The results are returned in a DataFrame.

    :param session: SQLAlchemy session for database access.
    :param layer_id: The ID of the layer to retrieve the year loss data for.
    :return: A DataFrame containing the year loss data for the specified layer.
    """
    query = select(LayerYearLoss).filter_by(layer_id=layer_id)
    return pd.read_sql_query(query, session.connection())


def get_df_loss(
    df_layeryearloss: pd.DataFrame, simulated_years: int, statistic: str
) -> pd.DataFrame:
    """
    Calculate the loss data for a specified statistic from the layer year loss data.

    This function processes the provided layer year loss DataFrame to calculate the loss data
    for either Aggregate Exceedance Probability (AEP) or Occurrence Exceedance Probability (OEP).
    The results are returned in a DataFrame with the specified statistic and corresponding percentiles.

    :param df_layeryearloss: DataFrame containing the layer year loss data.
    :param simulated_years: The number of simulated years for the analysis.
    :param statistic: The type of statistic to calculate, "AEP" for Aggregate Exceedance Probability
                      or "OEP" for Occurrence Exceedance Probability.
    :raises ValueError: If the statistic is not "AEP" or "OEP".
    :return: A DataFrame containing the calculated loss data for the specified statistic.
    """
    if statistic not in ["AEP", "OEP"]:
        raise ValueError("Statistic must be either 'AEP' or 'OEP'")

    # Determine the aggregation function based on the statistic type
    agg_func = "sum" if statistic == "AEP" else "max"

    # Aggregate the ceded losses by year
    ceded_by_year = df_layeryearloss.groupby("year")["ceded"].agg(agg_func)

    # Complete ceded_by_year with years with zero losses
    ceded_by_year_completed = ceded_by_year.reindex(
        range(simulated_years), fill_value=0
    )

    # Create a DataFrame for the specified statistic and percentiles
    df_loss = pd.DataFrame({"statistic": statistic, "percentile": PERCENTILES})

    # Calculate the loss for each percentile
    df_loss["loss"] = df_loss["percentile"].map(
        lambda x: int(ceded_by_year_completed.quantile(x, interpolation="higher"))
    )

    return df_loss
