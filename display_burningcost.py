import sys
from pathlib import Path

from win32com.client import Dispatch

from engine.function_burningcost import get_table_layerburningcost
from utils import df_from_listobject

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    wb = excel.Workbooks.Open(f"{Path.cwd()}/display_burningcost.xlsm")

# --------------------------------------
# Step 2: Read the input data
# --------------------------------------

ws_input = wb.Worksheets("Input")
layer_id = ws_input.Range("layer_id").value
df_layerburningcost = df_from_listobject(ws_input.ListObjects("layerburningcost"))

# --------------------------------------
# Step 3:   Process the burning cost
#           for the selected layer
# --------------------------------------

table_layerburningcost = get_table_layerburningcost(df_layerburningcost)

# --------------------------------------
# Step 4: Write the output data
# --------------------------------------

# Define the output worksheet and table
ws_output = wb.Worksheets("Output")
table_output = ws_output.ListObjects("table_layerburningcost")

# Clear the output table
if table_output.DataBodyRange is None:
    pass
else:
    table_output.DataBodyRange.Delete()

# Define the range for writing the output data, then write
cell_start = table_output.Range.Cells(2, 1)
cell_end = table_output.Range.Cells(2, 1).Offset(
    len(table_layerburningcost), len(table_layerburningcost.columns)
)
ws_output.Range(cell_start, cell_end).Value = table_layerburningcost.values
ws_output.Select()
