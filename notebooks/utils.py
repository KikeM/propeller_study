def load_polar(path, file_lift, file_drag, reader, airfoil_type, which="cl"):
    """Loads the airfoil polars
    
    Parameters
    ----------
    path : Path-like
    file_lift : str
    file_drag : str
    reader : csv-reader object
    airfoil_type : str
    which : str
        "cl" or "cd"
    
    Returns
    -------
    np.array N x 2
    """

    _path_lift = path / airfoil_type / file_lift
    _path_drag = path / airfoil_type / file_drag

    polar_cl = reader(_path_lift).values
    polar_cd = reader(_path_drag).values

    polars = {"cl": polar_cl, "cd": polar_cd}

    return polars[which]