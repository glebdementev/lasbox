import os
import time
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import laspy
import numpy as np
import torch

# Filtering
from modules.filter.componentFilter import filterPoints

# Segmentation (TreeisoNet)
from modules.treeisonet.treeLoc import treeLoc as treeLoc_infer
from modules.treeisonet.treeLoc import postPeakExtraction
from modules.treeisonet.treeOff import treeOff as treeOff_infer


class FilterModels(Enum):
    TREEFilter = "treefiltering_als_esegformer3D_128_15cm(GPU3GB)"


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


def resolve_paths(models_dir: Path) -> Tuple[Path, Path, Path, Path, Path, Path]:
    repo_dir = Path(__file__).parent

    # Filtering
    filter_name = FilterModels.TREEFilter.value
    filter_config = repo_dir / "modules" / "filter" / f"{filter_name}.json"
    filter_model = models_dir / f"{filter_name}.pth"

    # Segmentation
    treeloc_name = ReclamationModels.TREELoc.value
    treeoff_name = ReclamationModels.TREEOff.value
    treeloc_config = repo_dir / "modules" / "treeisonet" / f"{treeloc_name}.json"
    treeoff_config = repo_dir / "modules" / "treeisonet" / f"{treeoff_name}.json"
    treeloc_model = models_dir / f"{treeloc_name}.pth"
    treeoff_model = models_dir / f"{treeoff_name}.pth"

    assert filter_config.exists()
    assert filter_model.exists()
    assert treeloc_config.exists()
    assert treeoff_config.exists()
    assert treeloc_model.exists()
    assert treeoff_model.exists()

    return filter_config, filter_model, treeloc_config, treeloc_model, treeoff_config, treeoff_model


def read_las_xyz(las_path: Path) -> np.ndarray:
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    pcd = np.stack((x, y, z), axis=1)
    return pcd


def main() -> None:
    las_path_input = input("Enter path to LAS file: ").strip()
    las_path = Path(las_path_input)
    assert las_path.exists()

    print("\n=== Initializing configuration ===")
    cfg = default_run_config()
    print(f"Models directory: {cfg.models_dir}")

    print("\n=== Resolving model paths ===")
    filter_config, filter_model, treeloc_config, treeloc_model, treeoff_config, treeoff_model = resolve_paths(cfg.models_dir)
    print(f"Filter config: {filter_config}")
    print(f"Filter model: {filter_model}")
    print(f"TreeLoc config: {treeloc_config}")
    print(f"TreeLoc model: {treeloc_model}")
    print(f"TreeOff config: {treeoff_config}")
    print(f"TreeOff model: {treeoff_model}")

    print("\n=== Reading LAS file ===")
    start_time = time.time()
    pcd = read_las_xyz(las_path)
    n_pts_total = int(pcd.shape[0])
    assert n_pts_total > 0
    read_time = time.time() - start_time
    print(f"Loaded {n_pts_total:,} points in {read_time:.2f} seconds")

    print("\n=== Step 1: TreeFiltering ===")
    start_time = time.time()
    preds_filter = filterPoints(
        str(filter_config),
        pcd,
        str(filter_model),
        if_bottom_only=False,
        use_efficient=("esegformer" in FilterModels.TREEFilter.value.lower()),
        use_cuda=cfg.use_gpu,
    )
    filter_time = time.time() - start_time
    print(f"TreeFiltering completed in {filter_time:.2f} seconds")
    assert preds_filter is not None
    assert int(preds_filter.shape[0]) == n_pts_total
    preds_filter_i32 = preds_filter.astype(np.int32)

    keep_mask = preds_filter_i32 == 2
    kept_count = int(np.count_nonzero(keep_mask))
    dropped_count = int(preds_filter_i32.size - kept_count)
    dropped_share = (dropped_count / float(preds_filter_i32.size)) * 100.0 if preds_filter_i32.size > 0 else 0.0
    print(f"Kept points (trees): {kept_count:,}")
    print(f"Dropped points (non-trees): {dropped_count:,} ({dropped_share:.2f}%)")
    assert kept_count > 0, "No tree points retained after filtering."

    pcd_trees = pcd[keep_mask]
    n_pts_trees = int(pcd_trees.shape[0])

    custom_resolution = np.array([cfg.res_xy, cfg.res_xy, cfg.res_z], dtype=np.float32)
    print(f"Resolution: XY={cfg.res_xy}m, Z={cfg.res_z}m")

    print("\n=== Step 2: TreeLoc inference (on filtered points) ===")
    start_time = time.time()
    preds_conf_rad = treeLoc_infer(
        str(treeloc_config),
        pcd_trees,
        str(treeloc_model),
        use_cuda=cfg.use_gpu,
        if_stem=False,
        cutoff_thresh=cfg.cutoff_thresh,
        custom_resolution=custom_resolution,
    )
    treeloc_time = time.time() - start_time
    print(f"TreeLoc inference completed in {treeloc_time:.2f} seconds")
    assert preds_conf_rad is not None
    assert preds_conf_rad.shape[0] == n_pts_trees

    print("\n=== Filtering predictions (confidence) ===")
    start_time = time.time()
    conf_col = int(preds_conf_rad.shape[1] - 2)
    mask_conf = preds_conf_rad[:, conf_col] > cfg.conf_thresh
    filtered = preds_conf_rad[mask_conf]
    conf_filter_time = time.time() - start_time
    print(f"Filtered to {len(filtered):,} points (confidence > {cfg.conf_thresh}) in {conf_filter_time:.2f} seconds")
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

    print("\n=== Step 3: TreeOff inference (on filtered points) ===")
    start_time = time.time()
    labels = treeOff_infer(
        str(treeoff_config),
        pcd_trees,
        treeloc_tops,
        str(treeoff_model),
        use_cuda=cfg.use_gpu,
        custom_resolution=custom_resolution,
    )
    treeoff_time = time.time() - start_time
    print(f"TreeOff inference completed in {treeoff_time:.2f} seconds")
    assert labels is not None
    assert int(labels.shape[0]) == n_pts_trees

    print("\n=== Saving results (trees only with ITC) ===")
    start_time = time.time()
    las = laspy.read(str(las_path))
    out_las = laspy.LasData(las.header)
    out_las.points = las.points[keep_mask]

    # Save attributes: treefilter and pred_itc
    if "treefilter" not in out_las.point_format.extra_dimension_names:
        out_las.add_extra_dim(laspy.ExtraBytesParams(name="treefilter", type="int32", description="TreeFilter"))
    setattr(out_las, "treefilter", np.full((n_pts_trees,), 2, dtype=np.int32))

    preds_itc = labels.astype(np.int32)
    if "pred_itc" not in out_las.point_format.extra_dimension_names:
        out_las.add_extra_dim(laspy.ExtraBytesParams(name="pred_itc", type="int32", description="TreeisoNet ITC"))
    setattr(out_las, "pred_itc", preds_itc)

    out_path = las_path.with_name(f"{las_path.stem}_tree_itc.laz")
    out_las.write(str(out_path))
    save_time = time.time() - start_time
    print(f"Results saved to: {out_path}")
    print(f"Save completed in {save_time:.2f} seconds")

    print("\n=== Summary ===")
    total_time = read_time + filter_time + treeloc_time + conf_filter_time + peak_time + treeoff_time + save_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"  - Reading LAS: {read_time:.2f}s")
    print(f"  - TreeFiltering: {filter_time:.2f}s")
    print(f"  - TreeLoc inference: {treeloc_time:.2f}s")
    print(f"  - Confidence filtering: {conf_filter_time:.2f}s")
    print(f"  - Peak extraction: {peak_time:.2f}s")
    print(f"  - TreeOff inference: {treeoff_time:.2f}s")
    print(f"  - Saving: {save_time:.2f}s")


if __name__ == "__main__":
    main()


