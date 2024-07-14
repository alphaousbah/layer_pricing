import sys
from pathlib import Path

from win32com.client import Dispatch

from engine.function_layeryearloss import (
    get_table_layeryearloss_aep,
    get_table_layeryearloss_statistics,
)
from utils import df_from_listobject

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    wb = excel.Workbooks.Open(f"{Path.cwd()}/display_layeryearloss.xlsm")

# --------------------------------------
# Step 2: Read the input data
# --------------------------------------

ws_input = wb.Worksheets("Input")
resultinstance_id = ws_input.Range("resultinstance_id").value
simulated_years = int(ws_input.Range("simulated_years").value)
df_resultlayer = df_from_listobject(ws_input.ListObjects("ResultLayer"))
df_layeryearloss = df_from_listobject(ws_input.ListObjects("LayerYearLoss"))

# --------------------------------------
# Step 3: Display the result instance
# --------------------------------------

table_layeryearloss_statistics = get_table_layeryearloss_statistics(
    df_layeryearloss, df_resultlayer, simulated_years
)
table_layeryearloss_aep = get_table_layeryearloss_aep(
    df_layeryearloss, df_resultlayer, simulated_years
)

# --------------------------------------
# Step 4: Write the output data
# --------------------------------------

# Define the output worksheet and table
ws_output = wb.Worksheets("Output")

for df, listobject_name in [
    (table_layeryearloss_statistics, "table_layeryearloss_statistics"),
    (table_layeryearloss_aep, "table_layeryearloss_aep"),
]:
    table_output = ws_output.ListObjects(listobject_name)

    # Clear the output table
    if table_output.DataBodyRange is None:
        pass
    else:
        table_output.DataBodyRange.Delete()

    # Define the range for writing the output data, then write
    cell_start = table_output.Range.Cells(2, 1)
    cell_end = table_output.Range.Cells(2, 1).Offset(
        len(df),
        len(df.columns),
    )
    ws_output.Range(cell_start, cell_end).Value = df.values

ws_output.Select()
