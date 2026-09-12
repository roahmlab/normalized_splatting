# Normalized Gaussian Splatting

A fork of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
in which each Gaussian is normalized, allowing a probabilistic interpretation of rigid-body collisions as described in [Splanning](https://roahmlab.github.io/splanning/).

Turn it on with `--normalize`. Without that flag the rasterizer reproduces the original 3DGS method.

## Installation

Requires Python ≥ 3.10, CUDA, and a recent PyTorch.

```bash
git clone --recursive <repo-url>
cd normalized_splatting

# simple-knn needs a one-line <cfloat> include on current CUDA/GCC toolchains
git -C submodules/simple-knn apply ../../patches/simple-knn-cfloat.patch

pip install -r docker/requirements.txt
pip install -e .
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

For segmentation, optionally install the SAM and GroundingDINO submodules:

```bash
pip install -e submodules/segment-anything
pip install -e submodules/GroundingDINO   # needs a CUDA extension build
```

### Docker

```bash
./docker/build_base.sh && ./docker/build.sh
DATA_DIR=/path/to/your/data ./docker/run.sh
```

## Quickstart

```bash
# Train
python train.py -s <path_to_data> --normalize

# Render train/test views and compute metrics
python render.py -m <output_dir> --compute_metrics

# Or compute metrics separately
python -m evaluations.metrics -m <output_dir>
```

Outputs will be written to `./output/<timestamp>/` unless you pass `-m`.

## Datasets

The loader picks a reader by looking for a marker file in `--source_path`. We added a bunch more dataset loaders and have included them in the release in case it's useful. 

| Marker | Dataset |
|---|---|
| `sparse/` | COLMAP |
| `transforms_train.json` | Blender / NeRF-synthetic |
| `traj.txt` | Replica |
| `data.pickle` or `data.pickle.xz` | PyBullet |
| `poses.json` | Extracted RGB-D sequence (see `data_generation/`) |
| `groundtruth.txt` | TUM RGB-D |
| `kf_trajectory.txt` | ORB-SLAM |
| `dslr/` | ScanNet++ |

Converters between these and nerfstudio live in `normalized_splatting/data_generation/`.

## Options

### Normalization

| Flag | Default | Meaning |
|---|---|---|
| `--normalize` | off | Preset: enables normalization and sets the options below coherently |
| `--normalize_gaussians` | off | The rank-3 Jacobian and normalization constant |
| `--opacity_scale` | 1 / 0.001 | Global multiplier on opacity; normalized Gaussians produce much smaller raw alphas |
| `--opacity_scale_lr` | 0 | Learn the opacity scale (0 disables) |
| `--filter_radius_3d` | 0 / 1e-6 | Lower bound added to every 3D scale |
| `--filter_radius_2d` | 0.3 / 0.0 | Screen-space covariance dilation |

Where two defaults are listed, the second applies under `--normalize`. Any of
these can be set explicitly to override the preset.

### Depth, RGB-D and pose

| Flag | Default | Meaning |
|---|---|---|
| `--DS` | off | Depth supervision against sensor depth |
| `--depths` | `depths` | Depth subdirectory name |
| `--depth_trunc` | 1000 | Max sensor range; farther depths are ignored |
| `--min_depth` | none | Ignore depths nearer than this |
| `--lambda_depth` | 1.0 | Depth loss weight |
| `--BA` | off | Bundle adjustment: optimize camera poses jointly with the scene |
| `--localization` | off | Optimize poses only, freezing the Gaussians |
| `--pose_lr` | 0.1 | Initial pose learning rate |
| `--pose_lr_final` | 0.02 | Final pose learning rate |
| `--pose_lr_max_steps` | 300 | Pose schedule length |
| `--pose_trans_noise` | 0.0 | Inject translation noise into initial poses (for BA experiments) |
| `--single_frame_id` | none | Train against a single frame |

### Other

| Flag | Default | Meaning |
|---|---|---|
| `--app_opt` | off | Per-image appearance embedding + MLP colour head |
| `--scale_to_cube` / `--cube_scale` | off / 5.0 | Rescale the scene so cameras fit in a cube. Matters for normalization, whose constant is scale-dependent |
| `--image_stride` | 1 | Keep every Nth training image |
| `--fix_order` | off | Deterministic seeded camera order, for reproducible ablations |
| `--anisotropy_ratio` | 0 | Penalize Gaussians elongated beyond this max/min scale ratio (0 disables) |
| `--wandb` | off | Log to Weights & Biases (`wandb login` first) |

`build_all.py` trains many scenes across GPUs; `render_and_evaluate_all.sh`
renders and scores a directory of trained models.

## Reproducing the baselines

This section describes how to train the models reported in the SPLANNING paper. Note that some small bug fixes to the code have slightly changed the behavior from the released version. If you require a branch that exactly reproduces the paper experiments, please contact Seth Isaacson at sethgi@umich.edu. 

### Settings shared by both sets

Setting `--normalize` flips the following parameters to their normalized values:

| Option | Normalized | Unnormalized |
|---|---|---|
| `normalize_gaussians` | `True` | `False` |
| `opacity_scale` | `0.001` | `1` |
| `filter_radius_3d` | `1e-6` | `0` |
| `filter_radius_2d` | `0.0` | `0.3` |

### Replica and TUM RGB-D

```bash
REPLICA_SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
TUM_SCENES=(rgbd_dataset_freiburg1_360 rgbd_dataset_freiburg1_desk
            rgbd_dataset_freiburg1_desk2 rgbd_dataset_freiburg1_floor
            rgbd_dataset_freiburg1_plant rgbd_dataset_freiburg1_room
            rgbd_dataset_freiburg1_teddy rgbd_dataset_freiburg2_360_hemisphere
            rgbd_dataset_freiburg2_coke rgbd_dataset_freiburg2_dishes
            rgbd_dataset_freiburg2_flowerbouquet)

for scene in "${REPLICA_SCENES[@]}"; do
    python train.py -s "$REPLICA_BASE/$scene" -m "$OUT/unnormalized/$scene" --DS --eval
    python train.py -s "$REPLICA_BASE/$scene" -m "$OUT/normalized/$scene"   --normalize --DS --eval
done
```

and the same loop over `TUM_SCENES` under `$TUM_BASE`.

Then render and score:

```bash
./render_and_evaluate_all.sh -i 30000 "$OUT/normalized" "$OUT/unnormalized"
```

which writes `test/ours_30000/metrics.json` (SSIM, PSNR, render time, depth
RMSE) per scene.

### Simulation scenes

Simulation scenes are in DeebBlue: [https://deepblue.lib.umich.edu/data/concern/data_sets/c534fp99m](https://deepblue.lib.umich.edu/data/concern/data_sets/c534fp99m)/.

```bash
python build_all.py \
    -s /path/to/simulation_experiments/scenes \
    -m /path/to/models/normalized_3dgs \
    --normalize --DS
```

`build_all.py` walks `--scene_glob` (default `**/scene_*/`) under
`--source_path`, mirrors each scene's relative path under `--model_path`, and
runs one training process per GPU (`--gpu_ids` to restrict). Drop `--normalize`
for the unnormalized set. 


## Segmentation (SAGA)

Learns 32-D contrastive features per Gaussian, alpha-blended by the rasterizer,
so 3D objects can be selected by clicking or by text prompt. Run in order, after
training a scene:

```bash
# 1. SAM masks for every training view
python saga/extract_segment_everything_masks.py -m <output_dir> \
    --sam_checkpoint_path /path/to/sam_vit_h_4b8939.pth

# 2. 3D extent of each mask
python saga/get_scale.py -m <output_dir>

# 3. Train the contrastive features
python saga/train_contrastive_feature.py -m <output_dir>

# 4. Interactive viewer
python saga_gui.py -m <output_dir> -f <feature_iteration> -s <scene_iteration>
```

The viewer supports 2D and 3D click-to-segment, PCA and similarity views,
HDBSCAN clustering, rollback and per-object export. Text-prompted segmentation
is optional and needs a GroundingDINO checkpoint:

```bash
python saga_gui.py -m <output_dir> --grounding_dino_checkpoint /path/to/groundingdino_swint_ogc.pth
# or set $GROUNDING_DINO_CHECKPOINT
```

Without it the viewer runs normally and only the text prompt is disabled.

## Differences from vanilla 3DGS

Beyond the normalization itself, these defaults differ from upstream even
without `--normalize`:

- `position_lr_init` 5e-5 (upstream 1.6e-4)
- `opacity_lr` 5e-3 (upstream 5e-2)
- `scaling_lr` 1e-3 (upstream 5e-3)
- `reset_opacity()` is called once before the training loop
- Densification prunes at a screen-size threshold of 1 under `--normalize`
  (20 otherwise)

The rasterizer additionally returns depth, accumulated alpha and a 32-channel
contrastive feature image, all with gradients.

`submodules/diff-gaussian-rasterization/` is **not** a submodule — it is a
vendored fork of the Inria rasterizer, and it is where the normalization is
implemented (`cuda_rasterizer/forward.cu`, `backward.cu`). The symbolic
derivation of the backward pass is in `docs/derivations/gen_derivatives.py`.

## Acknowledgements

Built on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(Inria / MPII).

- The contrastive per-Gaussian feature field and the SAM mask pipeline in `saga/`
  follow **SAGA**.
- `saga_gui.py` is adapted from **OmniSeg3D-GS**, the Gaussian-Splatting port of
  **OmniSeg3D**, and inherits its hierarchical contrastive formulation.
- The scale-conditioned gate applied to those features follows **GARField**.
- The anisotropy regularizer (`--anisotropy_ratio`) follows **PhysGaussian**.

If you use this code, please cite the work it builds on:

```bibtex
@article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@ARTICLE{michauxisaacson2021splanning,
  author={Michaux, Jonathan and Isaacson, Seth and Adu, Challen Enninful and Li, Adam and Swayampakula, Rahul Kashyap and Ewen, Parker and Rice, Sean and Skinner, Katherine A. and Vasudevan, Ram},
  journal={IEEE Transactions on Robotics}, 
  title={Let us Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat}, 
  year={2025},
  volume={41},
  number={},
  pages={4380-4397},,
  doi={10.1109/TRO.2025.3584559}}

@inproceedings{cen2025saga,
  title     = {Segment Any 3D Gaussians},
  author    = {Cen, Jiazhong and Fang, Jiemin and Yang, Chen and Xie, Lingxi and
               Zhang, Xiaopeng and Shen, Wei and Tian, Qi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2025},
  eprint    = {2312.00860},
  archivePrefix = {arXiv}
}

@inproceedings{ying2024omniseg3d,
  title     = {OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning},
  author    = {Ying, Haiyang and Yin, Yixuan and Zhang, Jinzhi and Wang, Fan and
               Yu, Tao and Huang, Ruqi and Fang, Lu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  eprint    = {2311.11666},
  archivePrefix = {arXiv}
}

@inproceedings{kim2024garfield,
  title     = {GARField: Group Anything with Radiance Fields},
  author    = {Kim, Chung Min and Wu, Mingxuan and Kerr, Justin and Goldberg, Ken and
               Tancik, Matthew and Kanazawa, Angjoo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {21530--21539},
  year      = {2024},
  eprint    = {2401.09419},
  archivePrefix = {arXiv}
}

@inproceedings{xie2024physgaussian,
  title     = {PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics},
  author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and
               Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  eprint    = {2311.12198},
  archivePrefix = {arXiv}
}
```

Code: [SAGA](https://github.com/Jumpat/SegAnyGAussians) ·
[OmniSeg3D-GS](https://github.com/OceanYing/OmniSeg3D-GS) ·
[GARField](https://github.com/chungmin99/garfield) ·
[Segment Anything](https://github.com/facebookresearch/segment-anything) ·
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## License

This work inherits the original Gaussian-Splatting license held by Inria and MPII — see `LICENSE.md`. It is **free for non-commercial, research and evaluation use only**.