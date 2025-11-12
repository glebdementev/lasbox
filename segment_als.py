import os
import time
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
    use_gpu = bool(torch.cuda.is_available())
    if use_gpu:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        print(f"Using CUDA device: {device_name}")
    else:
        print("CUDA not available, using CPU")
    return RunConfig(
        models_dir=get_default_models_dir(),
        use_gpu=use_gpu,
        cutoff_thresh=0.0,
        conf_thresh=0.3,
        min_rad=0.2,
        max_gap=0.3,
        nms_thresh=0.5,
        # Match TreeAIBox default (UI sliders at 0 => use model config)
        res_xy=0.0,
        res_z=0.0,
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

    print("\n=== Initializing configuration ===")
    cfg = default_run_config()
    print(f"Models directory: {cfg.models_dir}")
    
    print("\n=== Resolving model paths ===")
    treeloc_config, treeloc_model, treeoff_config, treeoff_model = resolve_paths(cfg.models_dir)
    print(f"TreeLoc config: {treeloc_config}")
    print(f"TreeLoc model: {treeloc_model}")
    print(f"TreeOff config: {treeoff_config}")
    print(f"TreeOff model: {treeoff_model}")

    print("\n=== Reading LAS file ===")
    start_time = time.time()
    pcd = read_las_xyz(las_path)
    n_pts = int(pcd.shape[0])
    assert n_pts > 0
    read_time = time.time() - start_time
    print(f"Loaded {n_pts:,} points in {read_time:.2f} seconds")

    custom_resolution = np.array([cfg.res_xy, cfg.res_xy, cfg.res_z], dtype=np.float32)
    print(f"Resolution: XY={cfg.res_xy}m, Z={cfg.res_z}m")

    print("\n=== Step 1: TreeLoc inference ===")
    start_time = time.time()
    preds_conf_rad = treeLoc_infer(
        str(treeloc_config),
        pcd,
        str(treeloc_model),
        use_cuda=cfg.use_gpu,
        if_stem=False,
        cutoff_thresh=cfg.cutoff_thresh,
        custom_resolution=custom_resolution,
    )
    treeloc_time = time.time() - start_time
    print(f"TreeLoc inference completed in {treeloc_time:.2f} seconds")
    assert preds_conf_rad is not None
    assert preds_conf_rad.shape[0] == n_pts

    print("\n=== Filtering predictions ===")
    start_time = time.time()
    conf_col = int(preds_conf_rad.shape[1] - 2)
    mask = preds_conf_rad[:, conf_col] > cfg.conf_thresh
    filtered = preds_conf_rad[mask]
    filter_time = time.time() - start_time
    print(f"Filtered to {len(filtered):,} points (confidence > {cfg.conf_thresh}) in {filter_time:.2f} seconds")
    assert filtered.size > 0

    print("\n=== Peak extraction ===")
    start_time = time.time()
    treeloc_tops = postPeakExtraction(
        filtered,
        K=5,
        max_gap=cfg.max_gap,
        min_rad=cfg.min_rad,
        nms_thresh=cfg.nms_thresh,
    )
    peak_time = time.time() - start_time
    print(f"Extracted {len(treeloc_tops):,} tree tops in {peak_time:.2f} seconds")
    assert treeloc_tops is not None
    assert treeloc_tops.ndim == 2 and treeloc_tops.shape[1] >= 3

    print("\n=== Step 2: TreeOff inference ===")
    start_time = time.time()
    labels = treeOff_infer(
        str(treeoff_config),
        pcd,
        treeloc_tops,
        str(treeoff_model),
        use_cuda=cfg.use_gpu,
        custom_resolution=custom_resolution,
    )
    treeoff_time = time.time() - start_time
    print(f"TreeOff inference completed in {treeoff_time:.2f} seconds")
    assert labels is not None
    assert int(labels.shape[0]) == n_pts

    print("\n=== Saving results ===")
    start_time = time.time()
    las = laspy.read(str(las_path))
    preds = labels.astype(np.int32)
    if len(preds) == len(las.x):
        if "pred_itc" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="pred_itc", type="int32", description="TreeisoNet ITC"))
        las.pred_itc = preds
    out_path = las_path.with_name(f"{las_path.stem}_itc.laz")
    las.write(str(out_path))
    save_time = time.time() - start_time
    print(f"Results saved to: {out_path}")
    print(f"Save completed in {save_time:.2f} seconds")
    
    print("\n=== Summary ===")
    print(f"Total processing time: {read_time + treeloc_time + filter_time + peak_time + treeoff_time + save_time:.2f} seconds")
    print(f"  - Reading LAS: {read_time:.2f}s")
    print(f"  - TreeLoc inference: {treeloc_time:.2f}s")
    print(f"  - Filtering: {filter_time:.2f}s")
    print(f"  - Peak extraction: {peak_time:.2f}s")
    print(f"  - TreeOff inference: {treeoff_time:.2f}s")
    print(f"  - Saving: {save_time:.2f}s")


if __name__ == "__main__":
    main()


