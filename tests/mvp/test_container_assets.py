from pathlib import Path

from evalscope.config import parse_task_config
from evalscope.mvp.container_launcher import build_runtime_config

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_template_produces_a_loadable_mounted_fixture_task(tmp_path: Path) -> None:
    template = ROOT / 'configs' / 'mvp' / 'customer_multimodal_v1.yaml'
    original = template.read_bytes()

    runtime = build_runtime_config(template, tmp_path, 'custom_eval/multimodal/customer_v1', {})
    task = parse_task_config(str(runtime))

    assert task.datasets == ['customer_multimodal_v1']
    assert task.eval_type == 'mock_llm'
    assert task.model_id == 'text_generation'
    assert task.limit == 2
    assert task.eval_batch_size == 1
    assert task.work_dir == '/eval/output'
    assert task.no_timestamp is True
    dataset = task.dataset_args['customer_multimodal_v1']
    assert dataset['local_path'] == '/eval/data/custom_eval/multimodal/customer_v1'
    assert dataset['subset_list'] == ['example']
    fixture_dir = ROOT / Path(dataset['local_path']).relative_to('/eval/data')
    assert (fixture_dir / 'example.jsonl').is_file()
    assert template.read_bytes() == original
