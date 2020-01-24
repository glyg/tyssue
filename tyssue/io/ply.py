import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from ..version import version


from ..core.objects import Epithelium
from ..geomnetry.bulk_geometry import RNRGeometry
from ..topology import close_face


"""
PLY Header
----------

See also:

http://paulbourke.net/dataformats/ply/


.. code::
    # ply
    # format ascii 1.0
    # element vertex 3090
    # property float x
    # property float y
    # property float z
    # element face 6871
    # property list int int vertex_index
    # property float area
    # property int epidermis
    # element edge 9928
    # property int source
    # property int target
    # property list int int face_index
    # property float length
    # element volume 32
    # property list int int face_index
    # property int label
    # end_header

data types:
-----------

char       character                 1
uchar      unsigned character        1
short      short integer             2
ushort     unsigned short integer    2
int        integer                   4
uint       unsigned integer          4
float      single-precision float    4
double     double-precision float    8
"""

dtypes = {
    "char": np.uint8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}

dtypes_r = {np.dtype(v): k for k, v in dtypes.items()}
dtypes_r[np.dtype("int64")] = "int"
dtypes_r[np.dtype("uint64")] = "uint"


def _parse_header_line(line, element, structures, sizes):
    if line.startswith("element"):
        words = line.split(" ")
        element = words[1]
        structures[element] = {}
        sizes[element] = int(words[2])

    elif line.startswith("property"):
        words = line.split()
        if "list" in line:
            structures[element][words[-1]] = words[-3:-1]
        else:
            structures[element][words[-1]] = words[-2]
    return element


def parse_header(ply_f):
    structures, sizes = {}, {}
    element = ""
    sizes["header"] = 0
    for line in ply_f:
        sizes["header"] += 1
        if "end_header" in line:
            break
        element = _parse_header_line(line, element, structures, sizes)
    else:
        raise IOError("Header end string not found")
    offsets = np.cumsum(list(sizes.values()))
    offsets = {k: v for k, v in zip(list(sizes)[1:], offsets[:-1])}

    return structures, sizes, offsets


def _parse_line(line, structure):
    words = line.split()
    cursor = 0
    values = {}
    for key, dtype in structure.items():
        if isinstance(dtype, str):
            dtype = dtypes[dtype]
            word = words[cursor]
            # special case for boolean values
            if word == "True":
                values[key] = 1
            elif word == "False":
                values[key] = 0
            else:
                values[key] = dtype(words[cursor])
            cursor += 1
        elif isinstance(dtype, list):
            dtype = dtypes[dtype[1]]
            length = int(words[cursor])
            cursor += 1
            values[key] = [dtype(w) for w in words[cursor : cursor + length]]
            cursor += length
    return values


def _parse_generic(ply_f, structure, size, offset):
    ply_f.seek(0)
    df = pd.DataFrame(columns=structure.keys())
    for i, line in enumerate(ply_f):
        if i < offset:
            continue
        if i == size + offset:
            break
        df = df.append(_parse_line(line, structure), ignore_index=True)
    return df.dropna(axis=1, how="all")


def _parse_vertex(ply_f, structure, size, offset):
    ply_f.seek(0)
    df = pd.read_csv(ply_f, nrows=size, header=None, sep=" ", skiprows=offset)
    df.columns = structure.keys()

    return df


def _parse_tri_face(ply_f, structure, size, offset):
    """
    Triangular faces
    element face 6871
    property list int int vertex_index
    property float area
    property int epidermis
    """
    ply_f.seek(0)
    df = pd.read_csv(
        ply_f,
        nrows=size,
        sep=" ",
        header=None,
        usecols=[1, 2, 3, 4, 5],
        skiprows=offset,
    )
    df.columns = ["v0", "v1", "v2", "area", "epidermis"]

    return df


_parsers = {
    "vertex": _parse_vertex,
    "face": _parse_generic,
    "edge": _parse_generic,
    "volume": _parse_generic,
}


def read_ply(ply_file):

    with open(ply_file, "r", encoding="latin_1") as ply_f:
        structures, sizes, offsets = parse_header(ply_f)

        datasets = {}
        for element, structure in structures.items():

            parser = _parsers.get(element)
            if parser is None:
                continue
            size = sizes[element]
            offset = offsets[element]
            df = parser(ply_f, structure, size, offset)
            if element == "volume":
                datasets["cell"] = df
                df.index.name = "cell"
            else:
                datasets[element[:4]] = df
                df.index.name = element[:4]

    return datasets


def load_ply(ply_file):
    """Reads a ply file and reconstructs the datasets
    necessary to construct an epithelium.

    """

    datasets = read_ply(ply_file)

    # repeat each edge to have one half-edge per face
    n_faces = datasets["edge"]["face_index"].apply(len)
    edge_df_ = pd.DataFrame(
        np.repeat(datasets["edge"][["source", "target"]].to_numpy(), n_faces, axis=0),
        columns=["srce", "trgt"],
    )
    edge_df_["face_o"] = np.concatenate(datasets["edge"]["face_index"])

    for col in datasets["edge"].columns:
        if not col in ("source", "target", "face_index"):
            edge_df_[col + "_o"] = np.repeat(datasets["edge"][col].to_numpy(), n_faces)

    edge_df_["edge_o"] = np.repeat(datasets["edge"].index.to_numpy(), n_faces)

    # repeat each face to have one half-face per cell
    n_faces = datasets["cell"]["face_index"].apply(len)
    cell_faces = pd.DataFrame(
        np.repeat(datasets["cell"].index, n_faces), columns=["cell"]
    )
    cell_faces["face"] = np.concatenate(datasets["cell"]["face_index"])

    # for each edge, get the cells associated to its face
    edge_df_["cells"] = (
        cell_faces.groupby("face")["cell"]
        .apply(list)
        .reindex(edge_df_["face_o"])
        .to_numpy()
    )

    # when no cell is attributed to a cell, put -1
    edge_df_["cells"] = edge_df_["cells"].apply(
        lambda df: [-1] if not np.any(np.isfinite(df)) else df
    )

    # repeat the edges to have an half-edge for each face and cell
    # This is the correct number of edges
    edge_df = pd.DataFrame(
        np.repeat(edge_df_.to_numpy(), edge_df_["cells"].apply(len), axis=0),
        columns=edge_df_.columns,
    )

    edge_df["cell"] = np.concatenate(edge_df_["cells"].to_numpy())

    # clean up
    edge_df.drop(columns=["cells"], inplace=True)
    edge_df = edge_df.sort_values(["cell", "face_o"]).reset_index(drop=True)

    # face indices are still duplicated, we reindex the faces to avoid duplication
    # (frozensets are hashable so can be used as index)
    fc_pair = edge_df[["face_o", "cell"]].apply(frozenset, axis=1)

    ddup = fc_pair.drop_duplicates()
    new_faces = pd.Series(np.arange(ddup.shape[0]), index=ddup).reindex(fc_pair)
    edge_df["face"] = new_faces.to_numpy()

    cell_df = pd.DataFrame(index=edge_df["cell"].unique(), columns=list("xyz"))
    for col in datasets["cell"].columns:
        if col != "face_index":
            cell_df[col + "_o"] = datasets["cell"].loc[cell_df.index, col]

    face_df = pd.DataFrame(index=edge_df["face"].unique(), columns=list("xyz"))

    for col in datasets["face"].columns:
        if col != "vertex_index":
            edge_df["f_" + col] = (
                datasets["face"][col].take(edge_df["face_o"].to_numpy()).to_numpy()
            )
            face_df[col + "_o"] = edge_df.groupby("face")["f_" + col].first()
            edge_df.drop(columns=["f_" + col], inplace=True)

    return {"vert": datasets["vert"], "face": face_df, "edge": edge_df, "cell": cell_df}


def get_eptm(ply_file):
    mono = Epithelium("from_ply", load_ply(ply_file))
    RNRGeometry.update_all(mono)

    inverted_edges = mono.edge_df[mono.edge_df["sub_vol"] <= 0].index

    mono.edge_df.loc[inverted_edges, ["srce", "trgt"]] = mono.edge_df.loc[
        inverted_edges, ["trgt", "srce"]
    ].to_numpy()
    RNRGeometry.update_all(mono)
    RNRGeometry.center(mono)
    RNRGeometry.update_all(mono)

    mono.reset_index()
    mono.reset_topo()
    return mono


def _ply_header(eptm, columns, write_edges=True, write_cells=True):

    if eptm.Nv > 2 ** 32:
        raise OverflowError("Vertices are indexed on 2**32 bits only")

    lines = [
        "ply\n",
        "format ascii 1.0\n",
        f"comment file created by tyssue v.{version}\n",
        f"comment creation date: {datetime.today()}\n",
        f"comment epithelium {eptm.identifier}\n",
        f"element vertex {eptm.Nv}\n",
    ]
    # Vertices
    _append_cols(eptm, lines, "vert", columns)

    # Faces
    lines.append(f"element face {eptm.Nf}\n")
    lines.append("property list uchar int vertex_index\n")

    # Edges
    if write_edges:
        lines.append(f"element edge {eptm.Ne}\n")
        _append_cols(eptm, lines, "edge", columns)

    # Cells
    if write_cells and eptm.cell_df is not None:
        lines.append(f"element cell {eptm.Nc}\n")
        lines.append("property list uchar int vertex_index\n")

    lines.append("end_header\n")

    return lines


def _append_cols(eptm, lines, elem, columns):

    for col in columns[elem]:
        dtp = dtypes_r.get(eptm.datasets[elem].dtypes[col])
        if dtp is None:
            warnings.warn(
                f"""
{elem} data {col}'s data type, {eptm.datasets[elem].dtypes[col]} is not supported for PLY export, sorry.

Only numerical data types are supported. For booleans, please manually cast
your column to uint8.
            """
            )
        else:
            lines.append(f"property {dtp} {col}\n")


def _write_header(filename, eptm, columns, write_edges=True, write_cells=True):
    with open(filename, "w") as fh:
        fh.writelines(_ply_header(eptm, columns, write_edges, write_cells))


def _vertex_index_lists(eptm, elem):

    return (
        eptm.edge_df.groupby(elem)
        .apply(
            lambda df: f"{df[elem].iloc[0]} {' '.join((str(s) for s in df['srce']))}"
        )
        .to_list()
    )


def _write_elem_vertex_list(filename, eptm, elem):
    with open(filename, "a") as fh:
        fh.writelines(_vertex_index_lists(eptm, elem))


def _write_elem_data(filename, eptm, elem, columns):
    with open(filename, "a") as fh:
        eptm.datasets[elem].to_csv(fh, sep=" ", index=False, columns=columns[elem])


def write_ply(
    filename,
    eptm,
    extra_vert_cols=None,
    extra_edge_cols=None,
    write_edges=True,
    write_cells=True,
):

    columns = {"vert": set(eptm.coords), "edge": set(eptm.element_names)}

    if extra_vert_cols is not None:
        columns["vert"].update(extra_vert_cols)

    if extra_edge_cols is not None:
        columns["edge"].update(extra_edge_cols)

    _write_header(filename, eptm, columns, write_edges, write_cells)
    _write_elem_data(filename, eptm, "vert", columns)
    _write_elem_vertex_list(filename, eptm, "face")

    if write_edges:
        _write_elem_data(filename, eptm, "edge", columns)
    if write_cells:
        _write_elem_vertex_list(filename, eptm, "cell")
