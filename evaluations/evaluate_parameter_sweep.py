import argparse
import os
import re
import glob
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("output_dirs", nargs='+')
parser.add_argument("--results_dir", default="./sweep_metrics",
                    help="Where to write results_<iters>.csv (train/ and test/ subdirs).")
args = parser.parse_args()

print(args.output_dirs)
output_dirs = args.output_dirs

train_results = {}
test_results = {}

configs = {}

for output_dir in output_dirs:
    dataset_name = os.path.basename(os.path.normpath(output_dir))
    
    metrics_paths = glob.glob(f"{output_dir}/**/metrics.json", recursive=True)
    
    print(metrics_paths)
    
    for metric_path in metrics_paths:
        numbers = re.findall(r'\d+', metric_path)
        config_id, num_its = list(map(int, numbers))[-2:]
        
        test_train_str = os.path.dirname(metric_path).split(os.sep)[-2]
        
        config_path = f"{output_dir}/config_{config_id}/config.json"
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if config_id in configs:
            assert configs[config_id] == config_data
        else:
            configs[config_id] = config_data 
    
    
        with open(metric_path) as f:
            metrics = json.load(f)
        
        target = train_results if test_train_str == "train" else test_results
        
        if num_its not in target:
            target[num_its] = {}
            
        if config_id not in target[num_its]:
            target[num_its][config_id] = {}
            
            
        target[num_its][config_id][dataset_name] = metrics

def save_results(results, prefix):
    for num_its, all_configs in sorted(results.items()):
        records = []
        for config_id, datasets in sorted(all_configs.items()):
            record = {'config_id': config_id}
            # Add configuration parameters to the record
            config_data = configs[config_id]
            record.update(config_data)
            for dataset_name, metrics in datasets.items():
                for metric, value in metrics.items():
                    record[f"{metric} {dataset_name}"] = value
            records.append(record)
        
        df = pd.DataFrame(records)
        df = df.sort_values(by=['config_id'])
        os.makedirs(prefix, exist_ok=True)
        out_path = os.path.join(prefix, f'results_{num_its}.csv')
        df.to_csv(out_path, index=False)

        print(f"Results for {num_its} iterations saved to {out_path}")

if len(train_results) > 0:
    save_results(train_results, os.path.join(args.results_dir, "train"))

if len(test_results) > 0:
    save_results(test_results, os.path.join(args.results_dir, "test"))