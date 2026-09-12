#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

# This is a bit sloppy. For each named argument, if the command line argument is supplied, that value is used.
# Otherwise, if args.normalize is set, the SECOND value is used.
# otherwise, the first value is used (matching the original gaussian splatting params) 
# This will only work if each specified argument defaults to -1 in the original declaration
normalization_defaults = {
    "normalize_gaussians": [False, True],
    "opacity_scale":       [1, 0.001],
    "filter_radius_3d":    [0, 1e-6],
    "opacity_scale_lr":    [0, 0],        # don't change
    "filter_radius_2d":    [0.3, 0.0]
}

def set_normalization_defaults(args):
    """Resolve any option still at its -1 sentinel to the vanilla or normalized default.

    `normalize` is only defined by the training entry points; render.py reaches
    here with a Namespace rebuilt from a model's cfg_args, where every option is
    already resolved, so default to False rather than raising.
    """
    idx = int(bool(getattr(args, "normalize", False)))

    for argname in normalization_defaults:
        if not hasattr(args, argname) or getattr(args, argname) > -0.99:
            continue

        setattr(args, argname, normalization_defaults[argname][idx])


def check_normalization_requirements(args):
    """Reject --normalize without --DS, since the two are only calibrated together.

    Normalized splatting relies on depth supervision to constrain the geometry;
    without it the opacity/filtering defaults above have never been evaluated.
    --force-no-DS exists as an escape hatch for experiments, and only warns.
    """
    if not getattr(args, "normalize", False) or getattr(args, "DS", False):
        return

    if getattr(args, "force_no_DS", False):
        print("WARNING: --normalize was requested without --DS. Normalized splatting is "
              "only intended to be used with depth supervision, and this combination is "
              "untested -- I don't know what will happen. Continuing because --force-no-DS "
              "was passed.")
        return

    raise SystemExit(
        "--normalize requires --DS: normalized splatting is only intended to be used with "
        "depth supervision. Pass --DS, drop --normalize, or pass --force-no-DS to run "
        "anyway at your own risk.")


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = "depths"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        
        self.filter_radius_2d = -1.
        self.opacity_scale = -1.
        self.normalize_gaussians = -1.
        self.filter_radius_3d = -1.
        
        # Controls scaling down the scene. 
        self.scale_to_cube = False
        self.cube_scale = 5.
         
        self.feature_dim = 32
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00005
        self.position_lr_final = 0.0000005
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.005
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        # Camera-pose optimization (--BA / --localization). These are the values the
        # schedule in Camera.setup_optimizer has always effectively used.
        self.pose_lr = 0.1
        self.pose_lr_final = 0.02
        self.pose_lr_max_steps = 300
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth = 1.0
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.fix_order = False
        
        self.min_opacity = 0.005
        self.opacity_scale_lr = -1

        # Anisotropy regularizer (PhysGaussian); anisotropy_ratio = 0 disables it.
        self.lambda_anisotropy = 1.0
        self.anisotropy_ratio = 0.0

        # SAGA contrastive-feature training

        self.ray_sample_rate = 0
        self.num_sampled_rays = -1
        self.scale_aware_dim = -1
        self.rfn = 1.
        self.saga_feature_lr = 0.001

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser, args_str = None):
    if args_str is None:
        cmdlne_string = sys.argv[1:]
    else:
        cmdlne_string = args_str.replace("=", " ").split(" ")

    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)




    