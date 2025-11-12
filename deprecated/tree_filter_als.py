import os
import time
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import laspy
import numpy as np
import torch

# Pipeline module (TreeFiltering)
from modules.filter.componentFilter import filterPoints


class FilterModels(Enum):
    TREEFilter = "treefiltering_als_esegformer3D_128_15cm(GPU3GB)"


@dataclass
class RunConfig:
    models_dir: Path
    use_gpu: bool


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
    )


def resolve_paths(models_dir: Path) -> Tuple[Path, Path]:
    repo_dir = Path(__file__).parent
    model_name = FilterModels.TREEFilter.value
    config_path = repo_dir / "modules" / "filter" / f"{model_name}.json"
    model_path = models_dir / f"{model_name}.pth"
    assert config_path.exists()
    assert model_path.exists()
    return config_path, model_path


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
    config_path, model_path = resolve_paths(cfg.models_dir)
    print(f"Filter config: {config_path}")
    print(f"Filter model: {model_path}")

    print("\n=== Reading LAS file ===")
    start_time = time.time()
    pcd = read_las_xyz(las_path)
    n_pts = int(pcd.shape[0])
    assert n_pts > 0
    read_time = time.time() - start_time
    print(f"Loaded {n_pts:,} points in {read_time:.2f} seconds")

    print("\n=== TreeFiltering inference ===")
    start_time = time.time()
    preds = filterPoints(
        str(config_path),
        pcd,
        str(model_path),
        if_bottom_only=False,
        use_efficient=("esegformer" in FilterModels.TREEFilter.value.lower()),
        use_cuda=cfg.use_gpu,
    )
    infer_time = time.time() - start_time
    print(f"TreeFiltering completed in {infer_time:.2f} seconds")
    assert preds is not None
    assert int(preds.shape[0]) == n_pts

    print("\n=== Saving results ===")
    start_time = time.time()
    las = laspy.read(str(las_path))
    out_attr = "treefilter"
    preds_i32 = preds.astype(np.int32)

    # Drop non-tree points (class == 1), keep trees (class == 2)
    keep_mask = preds_i32 == 2
    kept_count = int(np.count_nonzero(keep_mask))
    dropped_count = int(preds_i32.size - kept_count)
    dropped_share = (dropped_count / float(preds_i32.size)) * 100.0 if preds_i32.size > 0 else 0.0
    print(f"Dropped points: {dropped_count} ({dropped_share:.2f}%)")

    # Build output LAS with only kept points
    out_las = laspy.LasData(las.header)
    out_las.points = las.points[keep_mask]

    # Write treefilter attribute for kept points (all 2s in binary model)
    kept_preds = preds_i32[keep_mask]
    if out_attr not in out_las.point_format.extra_dimension_names:
        out_las.add_extra_dim(laspy.ExtraBytesParams(name=out_attr, type="int32", description="TreeFilter"))
    setattr(out_las, out_attr, kept_preds)

    out_path = las_path.with_name(f"{las_path.stem}_{out_attr}.laz")
    out_las.write(str(out_path))
    save_time = time.time() - start_time
    print(f"Results saved to: {out_path}")
    print(f"Save completed in {save_time:.2f} seconds")

    print("\n=== Summary ===")
    total_time = read_time + infer_time + save_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"  - Reading LAS: {read_time:.2f}s")
    print(f"  - TreeFiltering: {infer_time:.2f}s")
    print(f"  - Saving: {save_time:.2f}s")


if __name__ == "__main__":
    main()
