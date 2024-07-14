# https://docs.sqlalchemy.org/en/20/orm/quickstart.html
# https://raaviblog.com/python-2-7-read-and-write-excel-file-with-win32com/
# https://learn.microsoft.com/en-us/dotnet/api/microsoft.office.interop.excel.listobject?view=excel-pia
import sys
from pathlib import Path
from time import perf_counter

import structlog
from win32com.client import Dispatch

from database import (
    Analysis,
    Layer,
    LayerYearLoss,
    ModelFile,
    ResultInstance,
    ResultLayer,
    ResultLayerReinstatement,
    ResultLayerStatisticLoss,
    Session,
)
from engine.function_layeryearloss import get_df_yearloss
from engine.function_resultlayerstatisticloss import get_df_resultlayerstatisticloss
from utils import (
    df_from_listobject,
    get_single_result,
    read_from_listobject_and_save,
    write_df_in_listobjects,
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
    wb = excel.Workbooks.Open(f"{Path.cwd()}/run_pricing.xlsm")

# --------------------------------------
# Step 2: Import the existing DB records
# --------------------------------------

with Session.begin() as session:
    read_from_listobject_and_save(
        session=session,
        worksheet=wb.Worksheets("Database"),
        listobject_names=[
            "Analysis",
            "Layer",
            "LayerReinstatement",
            "ModelFile",
            "ModelYearLoss",
        ],
    )

# --------------------------------------
# Step 3: Read the input data
# --------------------------------------

ws_input = wb.Worksheets("Input")

analysis_id = ws_input.Range("analysis_id").value
df_layer_modelfile = df_from_listobject(ws_input.ListObjects("layer_modelfile"))
simulated_years = int(ws_input.Range("simulated_years").value)

# --------------------------------------
# Step 4: Process
# --------------------------------------

start = perf_counter()


with Session.begin() as session:
    analysis = session.get(Analysis, analysis_id)

    if analysis is None:
        log.error(f"Analysis with id {analysis_id} not found.")
        raise ValueError(f"Analysis with id {analysis_id} not found")

    # Delete the previous relationships between layers and modelfiles
    for layer in analysis.layers:
        layer.modelfiles.clear()

    # Create and save the new relationships between layers and modelfiles
    for _, row in df_layer_modelfile.iterrows():
        layer_id = int(row["layer_id"])
        layer = get_single_result(session, Layer, layer_id)

        modelfile_id = int(row["modelfile_id"])
        modelfile: ModelFile = get_single_result(session, ModelFile, modelfile_id)

        layer.modelfiles.append(modelfile)

    # Calculate and save the layeryearlosses
    df_yearloss = get_df_yearloss(session, analysis_id, simulated_years)

    df_yearloss.to_sql(
        name="layeryearloss",
        con=session.connection(),
        if_exists="append",
        index=False,
    )

    # Create and save the resultinstance
    resultinstance = ResultInstance(name="Run 1")
    session.add(resultinstance)

    # Create the resultlayers
    layer_ids = df_layer_modelfile["layer_id"].unique()
    for layer_id in layer_ids:
        source_layer = get_single_result(session, Layer, layer_id)

        resultlayer = ResultLayer(
            occ_limit=source_layer.occ_limit,
            occ_deduct=source_layer.occ_deduct,
            agg_limit=source_layer.agg_limit,
            agg_deduct=source_layer.agg_deduct,
            source_id=source_layer.id,
        )
        resultinstance.layers.append(resultlayer)

        # Create and save the resultlayerreinstatements
        for layerreinstatement in layer.reinstatements:
            resultlayerreinstatement = ResultLayerReinstatement(
                order=layerreinstatement.order,
                number=layerreinstatement.number,
                rate=layerreinstatement.rate,
            )
            resultlayer.reinstatements.append(resultlayerreinstatement)

        # Create and save the relationships between resultlayers and modelfiles
        for modelfile in layer.modelfiles:
            resultlayer.modelfiles.append(modelfile)

        # Calculate and save the resultlayerstatisticlosses
        df_resultlayerstatisticloss = get_df_resultlayerstatisticloss(
            session, resultlayer.source_id, simulated_years
        )
        df_resultlayerstatisticloss.to_sql(
            name="resultlayerstatisticloss",
            con=session.connection(),
            if_exists="append",
            index=False,
        )

end = perf_counter()
print(f"Calculation time: {end - start}")

# --------------------------------------
# Step 5: Write the output data
# --------------------------------------

start = perf_counter()
ws_output = wb.Worksheets("Output")

with Session.begin() as session:
    write_df_in_listobjects(
        session=session,
        DbModels=[LayerYearLoss, ResultLayerStatisticLoss],
        ws_output=ws_output,
    )

ws_output.Select()
end = perf_counter()
print(f"Writing in Excel time: {end - start}")
