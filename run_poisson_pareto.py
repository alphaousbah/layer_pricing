import sys
from pathlib import Path
from time import perf_counter

import structlog
from win32com.client import Dispatch

from engine.function_poisson_pareto import get_poisson_pareto_ylt
from utils import (
    write_df_in_listobjects_simple,
)

log = structlog.get_logger()

# --------------------------------------
# Step 1: Open the Excel file
# --------------------------------------

excel = Dispatch("Excel.Application")

try:
    wb_path = sys.argv[1]
    wb = excel.Workbooks.Open(wb_path)
except IndexError:
    wb = excel.Workbooks.Open(f"{Path.cwd()}/run_poisson_pareto.xlsm")

# --------------------------------------
# Step 2: Read the input data
# --------------------------------------

ws_input = wb.Worksheets("Input")

poisson_rate = ws_input.Range("poisson_rate").value
pareto_shape = ws_input.Range("pareto_shape").value
pareto_scale = ws_input.Range("pareto_scale").value
no_year = int(ws_input.Range("no_year").value)

# --------------------------------------
# Step 3: Process
# --------------------------------------

start = perf_counter()
df_poisson_pareto_ylt = get_poisson_pareto_ylt(
    poisson_rate, pareto_shape, pareto_scale, no_year
)
end = perf_counter()
print(f"Calculation time: {end - start}")

# --------------------------------------
# Step 4: Write the output data
# --------------------------------------

start = perf_counter()
ws_output = wb.Worksheets("Output")
write_df_in_listobjects_simple(
    df_output=df_poisson_pareto_ylt,
    ws_output=ws_output,
    listobject_name="YearLossTable",
)
ws_output.Select()
end = perf_counter()
print(f"Writing in Excel time: {end - start}")
