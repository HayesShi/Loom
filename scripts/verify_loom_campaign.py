"""Check that the completed Loom reports match their saved run records."""
import json
import math
from pathlib import Path
import torch

from run_loom_campaign import ALL_TASKS, GRAPH_TASKS, configs, write_json
from loom_graph_campaign import configurations
from gcm.loom import Loom


def verify(output: Path):
    expected = [task for task in ALL_TASKS if task != 'laplace']
    expected += ['laplace_A', 'laplace_B', 'laplace_C']
    status = json.loads((output / 'campaign_status.json').read_text(encoding='utf-8'))
    assert status['total'] == len(expected) == 42
    assert not status['pending']
    assert set(status['completed']) == set(expected)
    current = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    assert not current['failed'], current['failed']
    checked = []
    for task in ALL_TASKS:
        folder = output / task
        assert not (folder / 'error.json').is_file(), task
        row = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        kind = 'binary' if row['primary'] in ['ROC_AUC', 'AP'] else ''
        candidates = configurations(task) if task in GRAPH_TASKS else configs(task, kind)
        records = []
        for index, cfg in enumerate(candidates):
            record = json.loads((folder / f'config_{index}.json').read_text(encoding='utf-8'))
            # Keep the actual batch in historical records when timing defaults change.
            expected_cfg = {key: value for key, value in cfg.items() if key != 'batch'}
            recorded_cfg = {key: value for key, value in record['config'].items() if key != 'batch'}
            assert recorded_cfg == expected_cfg, (task, index, 'configuration')
            assert math.isfinite(record['validation']), (task, index)
            assert (folder / f'config_{index}_best.pt').is_file(), (task, index)
            records.append(record)
        select = min if task in GRAPH_TASKS else max
        winner = select(records, key=lambda value: value['validation'])
        assert row['seed'] == 0, task
        assert row['config'] == winner['config'], (task, 'winner')
        assert row['best_epoch'] == winner['best_epoch'], (task, 'epoch')
        assert row['validation'] == winner['validation'], (task, 'validation')
        checkpoint = torch.load(folder / f"config_{winner['config_id']}_best.pt",
                                map_location='cpu', weights_only=False)
        assert checkpoint['seed'] == row['seed']
        assert checkpoint['config'] == row['config']
        assert checkpoint.get('best_epoch', checkpoint.get('epoch')) == row['best_epoch']
        assert checkpoint['val'] == row['validation']
        state = checkpoint['model']
        stages = len({name.split('.pairs.')[1].split('.')[0] for name in state
                      if name.startswith('blocks.0.pairs.')})
        node_capacity = row['nodes'] if task not in GRAPH_TASKS else 1 << stages
        model = Loom(state['self_projection.weight'].shape[1], state['head.weight'].shape[0],
                     node_capacity, **{key: row['config'][key]
                                       for key in ['dim', 'heads', 'blocks', 'dropout', 'angle']})
        model.load_state_dict(state, strict=True)
        del model, checkpoint, state
        assert row['training_speed_measurement'] == (1 if task in GRAPH_TASKS else 2), task
        if task in GRAPH_TASKS:
            assert row['preprocessing_measurement'] == 1, task
        for key, value in row['metrics'].items():
            assert math.isfinite(value), (task, key)
        for key in ['train_step_ms', 'inference_ms', 'peak_training_MiB', 'peak_inference_MiB']:
            assert math.isfinite(row[key]) and row[key] > 0, (task, key)
        if task in GRAPH_TASKS and 'logMSE' in row['metrics']:
            assert math.isclose(row['metrics']['logMSE'], math.log10(row['metrics']['MSE']),
                                rel_tol=1e-10, abs_tol=1e-10), task
        if task == 'laplace':
            for domain in ['A', 'B', 'C']:
                path = output / f'laplace_{domain}' / 'result.json'
                domain_row = json.loads(path.read_text(encoding='utf-8'))
                assert domain_row['config'] == row['config']
                assert domain_row['metrics'] == row['domains'][domain]
                assert (folder / f'{domain}_predictions.pt').is_file()
        else:
            assert (folder / 'test_predictions.pt').is_file(), task
        if task.startswith('reuse_'):
            assert all(key in row['metrics'] for key in ['Source_MSE', 'State_norm_ratio']), task
        if task == 'transfer_32':
            assert 0 <= row['metrics']['Effective_rank'] <= 256, task
        if task == 'plate':
            address = json.loads((folder / 'address_consistency.json').read_text(encoding='utf-8'))
            assert address['checkpoint'] == [winner['config_id'], winner['best_epoch']]
            assert address['near_nodes'] > 0 and address['far_nodes'] > 0
            for key in ['Hamming_distance', 'Path_agreement', 'Relabel_drift', 'Near_MSE', 'Far_MSE']:
                assert row['metrics'][key] == address['metrics'][key]
        checked.append({'task': task, 'configurations': len(records),
                        'selected_config': winner['config_id'], 'best_epoch': row['best_epoch']})
    roman = json.loads((output / 'roman_empire/result.json').read_text(encoding='utf-8'))
    execution = json.loads((output / 'execution_benchmark.json').read_text(encoding='utf-8'))
    assert execution['config'] == roman['config']
    assert len(execution['rows']) == 3
    for row in execution['rows']:
        assert math.isclose(row['Accuracy'], roman['metrics']['Accuracy'], abs_tol=1e-4)
        assert math.isfinite(row['Max_output_difference'])
    for name in ['RESULTS.md', 'results.csv', 'SPEED.md', 'speed.csv', 'execution_benchmark.csv',
                 'runtime.json', 'verification_tests.txt', 'plan.json']:
        assert (output / name).is_file(), name
    assert (output / 'source_snapshot/src/gcm/loom.py').is_file()
    payload = dict(state='complete', model='Loom', seed=0, tasks=len(ALL_TASKS),
                   result_rows=len(expected), checked=checked)
    write_json(output / 'verification.json', payload)
    return payload
