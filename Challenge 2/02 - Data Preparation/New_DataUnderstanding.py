import numpy as np
import pandas as pd
import os
from ydata_profiling import ProfileReport


def data_understanding(dataset):
    filename = "new_titanic_datasets/" + dataset
    df = pd.read_csv(filename)

    diretorio, _ = os.path.splitext(dataset)
    diretorio = str(diretorio) + " Profile Report"
    print(diretorio)

    if not os.path.exists(diretorio):
        os.mkdir(diretorio)

    html_file = diretorio + "/" + diretorio + ".html"
    print(html_file)
    if os.path.exists(html_file):
        os.remove(html_file)

    profile_report = ProfileReport(
        df,
        sort=None,
        html={
            "style": {"full_width": True}
        },
        progress_bar=True,
        correlations={
            "auto": {"calculate": True},
        },
        explorative=True,
        interactions={"continuous": True},
        title="Profiling Report"
    )

    profile_report.to_file(html_file)

for file in os.listdir('new_titanic_datasets'):
    data_understanding(file)

