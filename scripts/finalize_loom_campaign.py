"""Refresh graph preprocessing timings and save the completed campaign source."""
import json
from pathlib import Path
import shutil
import sys
import platform
import gc
import subprocess
import math

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts'))
import torch
from run_loom_campaign import ALL_TASKS,GRAPH_TASKS,configs,report,write_json
from loom_graph_campaign import configurations,benchmark_preprocessing,benchmark_graph_training
from gcm.loom import Loom


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    output=REPO/'results/loom'
    def incomplete_tuning():
        pending=[]
        for task in ALL_TASKS:
            count=len(configurations(task)) if task in GRAPH_TASKS else len(configs(task,''))
            if not (output/task/'result.json').is_file() or any(
                    not (output/task/f'config_{i}.json').is_file() for i in range(count)):
                pending.append(task)
            elif task not in GRAPH_TASKS:
                row=json.loads((output/task/'result.json').read_text(encoding='utf-8'))
                if row.get('training_speed_measurement')!=2:
                    pending.append(task)
        return pending
    pending=incomplete_tuning()
    remeasure_execution='roman_empire' in pending
    if pending:
        print('Supplementary tuning:',', '.join(pending),flush=True)
        subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_loom_campaign.py'),
                        '--retune','--tasks',*pending,'--output',str(output)],
                       cwd=REPO,check=True)
        pending=incomplete_tuning()
        if pending:
            raise RuntimeError(f'Unfinished tuning: {pending}')
    if remeasure_execution or not (output/'execution_benchmark.json').is_file():
        subprocess.run([sys.executable,'-u',str(REPO/'scripts/benchmark_loom_execution.py'),
                        '--output',str(output)],cwd=REPO,check=True)
    for task in GRAPH_TASKS:
        path=output/task/'result.json'
        if not path.is_file():
            continue
        row=json.loads(path.read_text(encoding='utf-8'))
        if 'MSE' in row['metrics'] and 'logMSE' in row['metrics']:
            row['metrics']['logMSE']=math.log10(max(row['metrics']['MSE'],1e-30))
            write_json(path,row)
        if row.get('preprocessing_measurement')!=1:
            timings=benchmark_preprocessing(task)
            row['dataset_preparation_seconds']=row['compile_seconds']
            row.update(timings,preprocessing_measurement=1)
            write_json(path,row)
            if task=='laplace':
                for domain in ['A','B','C']:
                    domain_path=output/f'laplace_{domain}'/'result.json'
                    domain_row=json.loads(domain_path.read_text(encoding='utf-8'))
                    domain_row.update(timings,preprocessing_measurement=1,
                                      dataset_preparation_seconds=row['dataset_preparation_seconds'])
                    write_json(domain_path,domain_row)
            print('Preprocessing timing:',task,flush=True)
        if row.get('training_speed_measurement')!=1:
            groups,info=torch.load(output/task/'packed.pt',weights_only=False)
            info['mean']=info['mean'].cuda();info['std']=info['std'].cuda()
            records=[json.loads(p.read_text(encoding='utf-8')) for p in (output/task).glob('config_*.json')]
            winner=min(records,key=lambda record:record['validation']);cfg=winner['config']
            model=Loom(info['input_dim'],info['output_dim'],info['max_nodes'],
                       **{k:cfg[k] for k in ['dim','heads','blocks','dropout','angle']}).cuda()
            checkpoint=torch.load(output/task/f"config_{winner['config_id']}_best.pt",weights_only=False)
            model.load_state_dict(checkpoint['model'])
            timings=benchmark_graph_training(model,groups,info,cfg)
            row.update(timings);write_json(path,row)
            if task=='laplace':
                for domain in ['A','B','C']:
                    domain_path=output/f'laplace_{domain}'/'result.json'
                    domain_row=json.loads(domain_path.read_text(encoding='utf-8'))
                    domain_row.update(timings);write_json(domain_path,domain_row)
            del model,groups,info,checkpoint
            gc.collect();torch.cuda.empty_cache()
            print('Training speed:',task,flush=True)
    subprocess.run([sys.executable,'-u',str(REPO/'scripts/measure_loom_plate_address.py'),
                    '--output',str(output)],cwd=REPO,check=True)
    report(output)
    snapshot=output/'source_snapshot'
    shutil.copytree(REPO/'src/gcm',snapshot/'src/gcm',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    script_names=['run_loom_campaign.py','loom_graph_campaign.py','loom_synthetic_data.py',
                  'benchmark_loom_execution.py','finalize_loom_campaign.py','measure_loom_plate_address.py',
                  'verify_loom_campaign.py',
                  'run_lrim.py','run_echo.py','run_plate_deformation.py','run_2dele.py',
                  'run_hodge_magnetostatics.py','run_sni_laplace.py','run_graphland.py']
    (snapshot/'scripts').mkdir(parents=True,exist_ok=True)
    for name in script_names:
        shutil.copy2(REPO/'scripts'/name,snapshot/'scripts'/name)
    (snapshot/'tests').mkdir(parents=True,exist_ok=True)
    for name in ['test_loom.py','test_lifting_pair.py','test_loom_capture_cuda.py']:
        shutil.copy2(REPO/'tests'/name,snapshot/'tests'/name)
    shutil.copy2(REPO/'pyproject.toml',snapshot/'pyproject.toml')
    write_json(output/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,
               cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=0,
               training_execution=2,model='gcm.loom.Loom',
               matmul_precision=torch.get_float32_matmul_precision()))
    write_json(output/'plan.json',dict(tasks=ALL_TASKS,seed=0,model='Loom',
               hardware=torch.cuda.get_device_name(),tuning='validation',
               configs_per_task={task:len(configurations(task)) if task in GRAPH_TASKS else len(configs(task,''))
                                 for task in ALL_TASKS}))
    validation=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_loom.py',
                               'tests/test_lifting_pair.py','tests/test_loom_capture_cuda.py'],
                              cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/'verification_tests.txt').write_text(validation.stdout,encoding='utf-8')
    print(validation.stdout,flush=True)
    validation.check_returncode()
    status=json.loads((output/'campaign_status.json').read_text(encoding='utf-8'))
    if status['pending']:
        raise RuntimeError(f"Unfinished datasets: {status['pending']}")
    if not (output/'execution_benchmark.json').is_file():
        raise RuntimeError('Execution benchmark is unfinished')
    from verify_loom_campaign import verify
    verify(output)
    print('Completed:',status['total'],'result rows',flush=True)


if __name__=='__main__':
    main()
