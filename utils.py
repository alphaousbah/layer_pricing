from typing import Type, TypeVar

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.orm import Session
from win32com.client import CDispatch

from database import Base

# Create a generic type T indicating that the function can return any type
T = TypeVar("T")


def df_from_listobject(listobject: CDispatch) -> pd.DataFrame:
    """
    Converts a ListObject to a pandas DataFrame.

    This function extracts data from a ListObject's DataBodyRange and HeaderRowRange,
    and uses them to create a pandas DataFrame.

    :param listobject: The ListObject to convert.
    :return: The resulting DataFrame.
    """
    data = listobject.DataBodyRange() if listobject.DataBodyRange else []
    columns = listobject.HeaderRowRange()
    return pd.DataFrame(data, columns=columns[0])


def read_from_listobject_and_save(
    session: Session, worksheet: CDispatch, listobject_names: list[str]
) -> None:
    """
    Reads data from list objects in a database, processes it, and saves it to an SQL database.

    This function iterates over a list of listobjects, converts each object to a pandas DataFrame,
    drops the 'id' column from the DataFrame, and saves the DataFrame to an SQL database using
    the SQLAlchemy engine.

    :param session: The SQLAlchemy Session for database operations.
    :param worksheet: The Excel Worksheet containing the Excel ListObjects.
    :param listobject_names: A list of Excel ListObjects to be read from the Worksheet.
    :return: None
    """
    for listobject_name in listobject_names:
        df = df_from_listobject(worksheet.ListObjects(listobject_name))
        df = df.drop(columns=["id"])
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        df.to_sql(
            name=str(listobject_name).lower(),
            con=session.connection(),
            if_exists="append",
            index=False,
        )
    return None


def write_df_in_listobjects_simple(
    df_output: pd.DataFrame, ws_output: CDispatch, listobject_name: str
) -> None:
    """
    Write the contents of a DataFrame into an Excel ListObject (table).

    :param df_output: The DataFrame containing the data to be written to the Excel table.
    :param ws_output: The Excel worksheet (as a CDispatch object) containing the ListObject.
    :param listobject_name: The name of the ListObject (table) in the worksheet.
    """
    table_output = ws_output.ListObjects(listobject_name)

    # Clear the output table
    if table_output.DataBodyRange is None:
        pass
    else:
        table_output.DataBodyRange.Delete()

    # Define the range for writing the output data, then write
    cell_start = table_output.Range.Cells(2, 1)
    cell_end = table_output.Range.Cells(2, 1).Offset(
        len(df_output), len(df_output.columns)
    )
    ws_output.Range(cell_start, cell_end).Value = df_output.values

    return None


def write_df_in_listobjects(
    session: Session, DbModels: list[Type[Base]], ws_output: CDispatch
) -> None:
    """
    Write data from database models to corresponding ListObjects in an Excel worksheet.

    This function retrieves data from the specified database models, clears the existing data
    in the corresponding ListObjects in the Excel worksheet, and writes the new data to them.

    :param session: SQLAlchemy session for database access.
    :param DbModels: List of database model classes to retrieve data from.
    :param ws_output: Excel worksheet object where data will be written.
    :return: None
    """
    for DbModel in DbModels:
        query = select(DbModel)
        df_output = pd.read_sql(query, session.connection())
        table_output = ws_output.ListObjects(DbModel.__name__)

        # Clear the output table
        if table_output.DataBodyRange is None:
            pass
        else:
            table_output.DataBodyRange.Delete()

        # Define the range for writing the output data, then write
        cell_start = table_output.Range.Cells(2, 1)
        cell_end = table_output.Range.Cells(2, 1).Offset(
            len(df_output), len(df_output.columns)
        )
        ws_output.Range(cell_start, cell_end).Value = df_output.values

    return None


def get_single_result(session: Session, DbModel: Type[T], object_id: int) -> T:
    """
    Execute a query to get a single result based on the object_id, raising an error if not found or multiple results are found.

    :param session: The SQLAlchemy session used to execute the query.
    :param DbModel: The SQLAlchemy ORM model to query, which is of type T (Type[T])
    :param object_id: The ID of the object to query.
    :return: The single result of the query which is an instance of type T.
    :raise ValueError: If no result is found, multiple results are found, or the model does not have an 'id' attribute.
    """
    if not hasattr(DbModel, "id"):
        raise ValueError(f"Model {DbModel.__name__} doesn't have an 'id' attribute")

    query = select(DbModel).where(DbModel.id == object_id)  # type: ignore[attr-defined]
    try:
        result = session.execute(query).scalar_one()
        if not isinstance(result, DbModel):
            raise ValueError(
                f"Expected instance of {DbModel.__name__}, got {type(result).__name__}"
            )
        return result
    except NoResultFound:
        raise ValueError(f"No result found for {DbModel.__name__} with id {object_id}")
    except MultipleResultsFound:
        raise ValueError(
            f"Multiple results found for {DbModel.__name__} with id {object_id}"
        )


"""

win32api.MessageBox(0, "Done", "Python")
ws_output.Select()

Other useful commands:

f = open('new_text_file.txt', 'x'
f.write('something')
f.close()
wb.Save()
wb.SaveAs('excelfile.xlsx')
wb.Close()
excel.Quit()
excel = None

"""
