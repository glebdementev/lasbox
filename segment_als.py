import os
from enum import Enum
from pathlib import Path
from typing import Optional

import laspy
import numpy as np
import torch
from dataclasses import dataclass

# Pipeline modules (TreeisoNet)
from modules.treeisonet.treeLoc import treeLoc as treeLoc_infer
from modules.treeisonet.treeLoc import postPeakExtraction
from modules.treeisonet.treeOff import treeOff as treeOff_infer


class ReclamationModels(Enum):
    TREELoc = "treeisonet_als_reclamation_treeloc_esegformer3D_128_10cm(GPU4GB)"
    TREEOff = "treeisonet_als_reclamation_treeoff_esegformer3D_128_10cm(GPU4GB)"


@dataclass
class RunConfig:
    models_dir: Path
    use_gpu: bool
    cutoff_thresh: float
    conf_thresh: float
    min_rad: float
    max_gap: float
    nms_thresh: float
    res_xy: float
    res_z: float


def get_default_models_dir() -> Path:
    if os.name == "nt":
        appdata_dir = Path(os.getenv("LOCALAPPDATA"))
    else:
        appdata_dir = Path.home() / ".local" / "share"
    model_dir = appdata_dir / "CloudCompare" / "TreeAIBox" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def default_run_config() -> RunConfig:
    return RunConfig(
        models_dir=get_default_models_dir(),
        use_gpu=bool(torch.cuda.is_available()),
        cutoff_thresh=0.0,
        conf_thresh=0.6,
        min_rad=0.2,
        max_gap=2.0,
        nms_thresh=1.0,
        res_xy=0.10,
        res_z=0.10,
    )


def resolve_paths(models_dir: Path) -> tuple[Path, Path, Path, Path]:
    repo_dir = Path(__file__).parent
    treeloc_name = ReclamationModels.TREELoc.value
    treeoff_name = ReclamationModels.TREEOff.value

    treeloc_config = repo_dir / "modules" / "treeisonet" / f"{treeloc_name}.json"
    treeoff_config = repo_dir / "modules" / "treeisonet" / f"{treeoff_name}.json"
    treeloc_model = models_dir / f"{treeloc_name}.pth"
    treeoff_model = models_dir / f"{treeoff_name}.pth"

    assert treeloc_config.exists()
    assert treeoff_config.exists()
    assert treeloc_model.exists()
    assert treeoff_model.exists()
    return treeloc_config, treeloc_model, treeoff_config, treeoff_model


def read_las_xyz(las_path: Path) -> np.ndarray:
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    pcd = np.stack((x, y, z), axis=1)
    return pcd


def write_csv(path: Path, data: np.ndarray, header: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        np.savetxt(path, data, delimiter=",")
        return
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main() -> None:
    las_path_input = input("Enter path to LAS file: ").strip()
    las_path = Path(las_path_input)
    assert las_path.exists()

    cfg = default_run_config()
    treeloc_config, treeloc_model, treeoff_config, treeoff_model = resolve_paths(cfg.models_dir)

    pcd = read_las_xyz(las_path)
    n_pts = int(pcd.shape[0])
    assert n_pts > 0

    custom_resolution = np.array([cfg.res_xy, cfg.res_xy, cfg.res_z], dtype=np.float32)

    # 1) TreeLoc inference
    preds_conf_rad = treeLoc_infer(
        str(treeloc_config),
        pcd,
        str(treeloc_model),
        use_cuda=cfg.use_gpu,
        if_stem=False,
        cutoff_thresh=cfg.cutoff_thresh,
        custom_resolution=custom_resolution,
    )
    assert preds_conf_rad is not None
    assert preds_conf_rad.shape[0] == n_pts

    conf_col = int(preds_conf_rad.shape[1] - 2)
    mask = preds_conf_rad[:, conf_col] > cfg.conf_thresh
    filtered = preds_conf_rad[mask]
    assert filtered.size > 0

    treeloc_tops = postPeakExtraction(
        filtered,
        K=5,
        max_gap=cfg.max_gap,
        min_rad=cfg.min_rad,
        nms_thresh=cfg.nms_thresh,
    )
    assert treeloc_tops is not None
    assert treeloc_tops.ndim == 2 and treeloc_tops.shape[1] >= 3

    # 2) TreeOff inference
    labels = treeOff_infer(
        str(treeoff_config),
        pcd,
        treeloc_tops,
        str(treeoff_model),
        use_cuda=cfg.use_gpu,
        custom_resolution=custom_resolution,
    )
    assert labels is not None
    assert int(labels.shape[0]) == n_pts

    # Save results into a LAS with an extra dimension "pred_itc"
    las = laspy.read(str(las_path))
    preds = labels.astype(np.int32)
    if len(preds) == len(las.x):
        if "pred_itc" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="pred_itc", type="int32", description="TreeisoNet ITC"))
        las.pred_itc = preds
    out_path = las_path.with_name(f"{las_path.stem}_itc.laz")
    las.write(str(out_path))
    print(str(out_path))


if __name__ == "__main__":
    main()


