# scm_configs.py
# Define dataset-specific SCM structures (nodes, parents, dims, and optional label maps for proxies)

from typing import Dict, List, Tuple

def get_scm_config(task_name: str):
    """
    Returns:
      nodes:     list of SCM node names used by NeuralSCM (these must match your model)
      parents:   list[List[int]] where parents[j] are indices of nodes that are direct parents of node j
      node_dims: list[int] per node (you can keep 1 or small dims per scalar proxy; tune as desired)
    """
    if task_name == "Blastocyst":
        nodes = ["ZPThicknessVar", "TEArea", "BubbleArea", "BlastocoelSym"]
        # ZPTV -> BlastocoelSym; TEArea -> BlastocoelSym; BubbleArea -> BlastocoelSym
        parents = [
            [],     # ZPThicknessVar
            [],     # TEArea
            [],     # BubbleArea
            [0,1,2] # BlastocoelSym depends on ZPThicknessVar, TEArea, BubbleArea
        ]
        node_dims = [1, 1, 1, 1]

    elif task_name == "HAM10K":
        nodes = ["LesionSize", "Compactness", "Asymmetry", "MeanRGB"]
        # LesionSize -> Compactness; LesionSize -> Asymmetry; MeanRGB -> Asymmetry
        parents = [
            [],     # LesionSize
            [0],    # Compactness depends on LesionSize
            [0,3],  # Asymmetry depends on LesionSize, MeanRGB
            []      # MeanRGB
        ]
        node_dims = [1, 1, 1, 1]

    elif task_name == "Fetal":
        nodes = ["HeadPerimeter", "HeadMajorAxis", "HeadArea", "HeadCircularity"]
        # HeadPerimeter -> HeadCircularity; HeadMajorAxis -> HeadCircularity; HeadArea -> HeadCircularity
        parents = [
            [],     # HeadPerimeter
            [],     # HeadMajorAxis
            [],     # HeadArea
            [0,1,2] # HeadCircularity depends on the three above
        ]
        node_dims = [1, 1, 1, 1]

    elif task_name == "Mosmed":
        nodes = ["LesionSize", "Solidity", "Orientation", "Perimeter"]
        # LesionSize -> Solidity; LesionSize -> Orientation; Perimeter -> Orientation
        parents = [
            [],     # LesionSize
            [0],    # Solidity depends on LesionSize
            [0,3],  # Orientation depends on LesionSize and Perimeter
            []      # Perimeter
        ]
        node_dims = [1, 1, 1, 1]

    elif task_name == "Kvasir":
        nodes = ["MeanRGB", "LesionSize", "Compactness", "Asymmetry"]
        # MeanRGB -> Asymmetry; LesionSize -> Compactness; LesionSize -> Asymmetry
        parents = [
            [],     # MeanRGB
            [],     # LesionSize
            [1],    # Compactness depends on LesionSize
            [1,0],  # Asymmetry depends on LesionSize and MeanRGB
        ]
        node_dims = [1, 1, 1, 1]

    else:
        # default (empty)
        nodes = []
        parents = []
        node_dims = []

    return {"nodes": nodes, "parents": parents, "node_dims": node_dims}