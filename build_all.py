import torch.multiprocessing as mp
import torch
import os, sys
import glob
import datetime

from argparse import ArgumentParser
from normalized_splatting.arguments import ModelParams, PipelineParams, OptimizationParams, set_normalization_defaults, check_normalization_requirements
from train import training, safe_state

# Used to tell
class StopSignal:
    pass

def training_function(source_path, model_path, args, lp, op, pp):
    args.source_path = source_path
    args.model_path = model_path
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args)

# Runs 
def _gpu_worker(gpu_id, fn_handle, job_queue: mp.Queue, result_queue: mp.Queue, *additional_args, **additional_kwargs):
    # Pin this worker to its GPU inside the child, before any CUDA context exists.
    # Setting it in the parent loop races with the next iteration.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while not job_queue.empty():
        data = job_queue.get()
        if isinstance(data, StopSignal):
            result_queue.put(StopSignal())
            break
        
        output_data = fn_handle(*data, *additional_args, **additional_kwargs)        
        result_queue.put(output_data)


def run_pool(fn_handle, job_data, gpu_ids = None, additional_args = (), additional_kwargs = {}):
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
       
    num_gpus = len(gpu_ids)
    
    # Load the jobs into the queue
    for args in job_data:
        job_queue.put(args)
        
    # One None corresponds to sending a stop signal to one of the workers
    for _ in range(num_gpus):
        job_queue.put(StopSignal())
        
    # Create and start the processes
    gpu_worker_processes = []
    for gpu_id in gpu_ids:
        gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(gpu_id, fn_handle, job_queue, result_queue, *additional_args), kwargs=additional_kwargs))
        gpu_worker_processes[-1].start()
    
    # Wait for everything to start
    stop_recv = 0
    results = []
    while stop_recv < num_gpus:
        result = result_queue.get()
        if isinstance(result, StopSignal):
            stop_recv += 1
            continue
        results.append(result)
    # Sync
    for process in gpu_worker_processes:
        process.terminate()
        
    return results

if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--pose_trans_noise', type=float, default=0.0)
    parser.add_argument("--DS", action="store_true", default=False)
    parser.add_argument("--BA", action="store_true", default=False)
    parser.add_argument("--localization", action="store_true", default=False)
    parser.add_argument('--single_frame_id', type=int, default=None)

    parser.add_argument("--depth_trunc", type=float, default=1000., help="for rgb-d datasets, sets the max range of the depth sensor.")
    parser.add_argument("--output_dir", type=str, default="./output", help="parent_dir for the output")
    parser.add_argument("--wandb", action='store_true')

    parser.add_argument("--normalize", action="store_true", help="If set, sets all settings for the default normalization"
                         "scheme (opacity_scale = 0.001, filter_radius_2d=0.0, filter_radius_3d=1e-6)")
    parser.add_argument("--force-no-DS", dest="force_no_DS", action="store_true", default=False,
                        help="Allow --normalize without --DS. Untested and unsupported.")
    parser.add_argument("--app_opt", action="store_true", help="Enable appearance optimization")
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    parser.add_argument("--scene_glob", type=str, default="**/scene_*/",
                        help="Glob, relative to --source_path, selecting the scene directories to train.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.test_iterations[-1])

    check_normalization_requirements(args)
    set_normalization_defaults(args)
    
    top_level_model_path = args.model_path or os.path.join("./output/", os.getenv('OAR_JOB_ID') or datetime.datetime.now().strftime("%m%d%y_%H%M%S"))
    
    job_data = []
    for world in sorted(glob.glob(args.scene_glob, root_dir=args.source_path, recursive=True)):
        scene_dir = os.path.join(args.source_path, world)
        if not os.path.isdir(scene_dir):
            continue
        model_path = os.path.join(top_level_model_path, world.strip("/"))
        print("Training on", scene_dir, "->", model_path)
        job_data.append((scene_dir, model_path))

    if not job_data:
        raise SystemExit(
            f"No scenes matched {args.scene_glob!r} under {args.source_path}")

    job_data.sort()
    
    results = run_pool(training_function, job_data, args.gpu_ids, (args, lp, op, pp))
