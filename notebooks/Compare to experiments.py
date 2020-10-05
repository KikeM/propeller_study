# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from functools import partial

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import pybem as pb

# %%
PATH_POLARS = "../polars"

# %%
csv_reader = partial(pd.read_csv, sep=";", index_col=None)


def load_polar(airfoil_type, which="cl"):

    _path_lift = Path(PATH_POLARS) / airfoil_type / "lift.csv"
    _path_drag = Path(PATH_POLARS) / airfoil_type / "drag.csv"

    polar_cl = csv_reader(_path_lift).values
    polar_cd = csv_reader(_path_drag).values

    polars = {"cl": polar_cl, "cd": polar_cd}

    return polars[which]


# %%
# Define known sections
sections = [
    pb.Section(
        name="Hub",
        r=0.3,
        beta=60,
        chord=0.4,
        airfoil=pb.Airfoil(
            polar_cl=load_polar(airfoil_type="765", which="cl"),
            polar_cd=load_polar(airfoil_type="765", which="cd"),
        ),
    ),
    pb.Section(
        name="Station 0",
        r=0.3,
        beta=60,
        chord=0.4,
        airfoil=pb.Airfoil(
            polar_cl=load_polar(airfoil_type="765", which="cl"),
            polar_cd=load_polar(airfoil_type="765", which="cd"),
        ),
    ),
    pb.Section(
        name="Station 1",
        r=0.6,
        beta=45,
        chord=0.35,
        airfoil=pb.Airfoil(
            polar_cl=load_polar(airfoil_type="764", which="cl"),
            polar_cd=load_polar(airfoil_type="764", which="cd"),
        ),
    ),
    pb.Section(
        name="Station 2",
        r=0.6,
        beta=45,
        chord=0.35,
        airfoil=pb.Airfoil(
            polar_cl=load_polar(airfoil_type="763", which="cl"),
            polar_cd=load_polar(airfoil_type="763", which="cd"),
        ),
    ),
    pb.Section(
        name="Tip",
        r=1.2,
        beta=30,
        chord=0.2,
        airfoil=pb.Airfoil(
            polar_cl=load_polar(airfoil_type="762", which="cl"),
            polar_cd=load_polar(airfoil_type="762", which="cd"),
        ),
    ),
]

# Define propeller
B = 6
propeller = pb.Propeller(B=B, sections=sections)

# Define flow conditions and BEM method
J = 0.2
bem = pb.BladeElementMethod(J=J, propeller=propeller, tip_loss=False, hub_loss=False)

# Solve
bem.solve()

# Compute forces
CT, CQ = bem.integrate_forces()

# %%
