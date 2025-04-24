#!/usr/bin/env python3
"""
TensorRT 10.9 inference for NVIDIA FoundationStereo
with disparity visualisation identical to the PyTorch utility
and coloured point-cloud export.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import imageio
import tensorrt as trt
import time
import pycuda.driver as cuda
import pycuda.autoinit                       # noqa:  F401
import open3d as o3d

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ───────────────────────── TensorRT helpers ──────────────────────────
def load_engine(engine_path: Path) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f:
        rt = trt.Runtime(TRT_LOGGER)
        return rt.deserialize_cuda_engine(f.read())


def list_io_tensors(engine: trt.ICudaEngine):
    ios = []
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        mode  = engine.get_tensor_mode(name)
        ios.append(dict(idx=i, name=name, shape=shape, is_in=(mode == trt.TensorIOMode.INPUT)))
    return ios


def volume(shape: tuple[int, ...]) -> int:
    v = 1
    for d in shape:
        v *= d
    return v


# ───────────────────────── image utilities ───────────────────────────
def resize_and_pad(path: Path, target_hw: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """
    Resize an image preserving aspect ratio, then center-pad to target (H,W).
    Returns the padded RGB uint8 image and a tfm dict (scale_x/y, pad_x/y).
    """
    im_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    print(f"[TRT] original   : {im_bgr.shape}") 
    if im_bgr is None:
        raise FileNotFoundError(path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    H_t, W_t = target_hw
    h, w     = im_rgb.shape[:2]
    scale    = (W_t / w) if w > h else (H_t / h)
    new_w, new_h = int(w * scale), int(h * scale)

    im_rs  = cv2.resize(im_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    print(f"[TRT] after resize: {im_rs.shape}")
    top    = (H_t - new_h) // 2
    left   = (W_t - new_w) // 2
    pad_bot = H_t - new_h - top
    pad_rgt = W_t - new_w - left

    im_pad = cv2.copyMakeBorder(im_rs, top, pad_bot, left, pad_rgt,
                             borderType=cv2.BORDER_REPLICATE)
    print(f"[TRT] after pad   : {im_pad.shape}")
    tfm = dict(scale_x=scale, scale_y=scale, pad_x=left, pad_y=top)
    return im_pad, tfm


def to_blob(img: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB → float32 NCHW in [0,1] contiguous."""
    blob = img.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob)


# ───────────────────── disparity visualisation ───────────────────────
def vis_disparity(disp: np.ndarray) -> np.ndarray:
    """
    Emulates `core.utils.utils.vis_disparity()` from the PyTorch repo.
    Steps:
      • replace non-finite with zero
      • compute percentile-95 of positive values for robust max
      • normalise → uint8 0-255
      • apply JET colour-map, return RGB uint8
    """
    disp_v = disp.copy()
    disp_v[~np.isfinite(disp_v)] = 0
    pos = disp_v > 0
    if not pos.any():
        return np.zeros((*disp_v.shape, 3), dtype=np.uint8)

    v_min = disp_v[pos].min()
    v_max = np.percentile(disp_v[pos], 95)
    disp_norm = np.clip((disp_v - v_min) / (v_max - v_min + 1e-6), 0, 1)
    disp_u8   = (disp_norm * 255).astype(np.uint8)
    disp_col  = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(disp_col, cv2.COLOR_BGR2RGB)


# ───────────────────────── geometry helpers ──────────────────────────
def adjust_intrinsics(K: np.ndarray, tfm: dict) -> np.ndarray:
    K = K.copy()
    K[0, 0] *= tfm["scale_x"]
    K[1, 1] *= tfm["scale_y"]
    K[0, 2] = K[0, 2] * tfm["scale_x"] + tfm["pad_x"]
    K[1, 2] = K[1, 2] * tfm["scale_y"] + tfm["pad_y"]
    return K


def depth_to_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (x - K[0, 2]) * z / K[0, 0]
    y = (y - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


# ───────────────────────────────── main ───────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine",          default="output/foundation_stereo.engine", type=Path)
    p.add_argument("--left",            default="assets/left.jpg",                 type=Path)
    p.add_argument("--right",           default="assets/right.jpg",                type=Path)
    p.add_argument("--intrinsic_file",  default="assets/K.txt",                    type=Path)
    p.add_argument("--out",             default="trt_output",                      type=Path)
    p.add_argument("--z_far",           default=10.0, type=float)
    p.add_argument("--denoise",         action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    engine  = load_engine(args.engine)
    ctx     = engine.create_execution_context()

    ios = list_io_tensors(engine)
    ins  = [t for t in ios if t["is_in"]]
    outs = [t for t in ios if not t["is_in"]]
    nameL, shapeL = ins[0]["name"], tuple(ins[0]["shape"])
    nameR, shapeR = ins[1]["name"], tuple(ins[1]["shape"])
    nameO, shapeO = outs[0]["name"], tuple(outs[0]["shape"])

    d_buf = {}
    for t in ios:
        dptr = cuda.mem_alloc(volume(t["shape"]) * 4)
        d_buf[t["name"]] = dptr
        ctx.set_tensor_address(t["name"], int(dptr))

    TARGET_HW = (shapeL[2], shapeL[3])               # (H,W)
    print("TARGET_HW:", TARGET_HW)
    left_img,  tfm = resize_and_pad(args.left,  TARGET_HW)
    right_img, _   = resize_and_pad(args.right, TARGET_HW)

    blobL = to_blob(left_img)
    blobR = to_blob(right_img)
    print(f"[TRT] blob shape  : {blobL.shape} (N,C,H,W)")

    stream = cuda.Stream()
    def copy_in():
        cuda.memcpy_htod_async(int(d_buf[nameL]), blobL, stream)
        cuda.memcpy_htod_async(int(d_buf[nameR]), blobR, stream)

    # for _ in range(10):               # warm-up
    #     copy_in(); ctx.execute_async_v3(stream.handle); stream.synchronize()

    copy_in()

    t0 = time.perf_counter()         # ← start timer
    ctx.execute_async_v3(stream.handle)
    stream.synchronize()
    t1 = time.perf_counter()         # ← stop  timer

    h_out = np.empty(shapeO, dtype=np.float32)
    cuda.memcpy_dtoh(h_out, d_buf[nameO])
    print(f"✓ inference time: {(t1 - t0)*1e3:.2f} ms "
        f"({1/(t1 - t0):.1f} FPS)")
    
    disp = h_out[0, 0] if h_out.ndim == 4 else h_out[0]

    # ───── disparity visualisation (identical to PyTorch helper) ─────
    disp_vis = vis_disparity(disp)
    imageio.imwrite(args.out / "vis.png",
                    np.concatenate([left_img, disp_vis], axis=1))
    print("✓ disparity visualisation →", args.out / "vis.png")

    # ───── depth + point cloud ─────
    K, baseline = np.loadtxt(args.intrinsic_file, max_rows=1).reshape(3, 3), float(open(args.intrinsic_file).read().splitlines()[1])
    K = adjust_intrinsics(K.astype(np.float32), tfm)

    valid  = np.isfinite(disp) & (disp > 0)
    depth  = np.full_like(disp, np.nan, dtype=np.float32)
    depth[valid] = K[0, 0] * baseline / disp[valid]

    xyz     = depth_to_xyz(depth, K)
    mask    = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0) & (xyz[..., 2] <= args.z_far)
    mask_f  = mask.ravel()
    xyz_f   = xyz.reshape(-1, 3)
    col_f   = left_img.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_f[mask_f])
    pcd.colors = o3d.utility.Vector3dVector(col_f[mask_f])
    o3d.io.write_point_cloud(str(args.out / "cloud.ply"), pcd)
    print("✓ raw point-cloud          →", args.out / "cloud.ply")

    if args.denoise:
        _, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        pcd_dl = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(str(args.out / "cloud_denoise.ply"), pcd_dl)
        print("✓ denoised point-cloud     →", args.out / "cloud_denoise.ply")

    np.save(args.out / "depth_meter.npy", depth)
    print("✓ depth map (metres)       →", args.out / "depth_meter.npy")


if __name__ == "__main__":
    main()
