from pathlib import Path
import csv
import numpy as np

# Paths adapted from the user's CALVIN scripts
TRAIN_DIR = Path('/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D/validation')
ANN_PATH = TRAIN_DIR / 'lang_annotations' / 'auto_lang_ann.npy'
OUT_CSV = Path('/home/yewon/RoboVLMs/calvin_task_episode_instruction_matches_val.csv')

TARGET_TASKS = {
    'lift_blue_block_table',
    'lift_pink_block_table',
    'lift_red_block_table',
    'place_in_drawer',
}

# If you only want Scene D like in your second script, leave this enabled.
# If not, set FILTER_SCENE_D = False.
FILTER_SCENE_D = True
SCENE_D_START = 0
SCENE_D_END = 611098


def normalize_text(x):
    if isinstance(x, bytes):
        return x.decode('utf-8')
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return str(x.item())
        return ' '.join(map(str, x.tolist()))
    return str(x)


def load_annotations(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Annotation file not found: {path}')

    data = np.load(path, allow_pickle=True).item()
    tasks = data['language']['task']
    anns = data['language']['ann']
    indx = data['info']['indx']
    emb = data['language'].get('emb', None)

    if not (len(tasks) == len(anns) == len(indx)):
        raise ValueError(
            f'Length mismatch: task={len(tasks)}, ann={len(anns)}, indx={len(indx)}'
        )

    return tasks, anns, indx, emb


def in_scene_d(start: int, end: int) -> bool:
    return SCENE_D_START <= start and end <= SCENE_D_END


def main():
    tasks, anns, indx, emb = load_annotations(ANN_PATH)

    rows = []
    for seg_id, (task, ann, seg) in enumerate(zip(tasks, anns, indx)):
        task = normalize_text(task)
        if task not in TARGET_TASKS:
            continue

        start, end = int(seg[0]), int(seg[1])
        if FILTER_SCENE_D and not in_scene_d(start, end):
            continue

        instruction = normalize_text(ann)
        frame_count = end - start + 1

        rows.append({
            'seg_id': seg_id,
            'task': task,
            'instruction': instruction,
            'start_frame': start,
            'end_frame': end,
            'frame_count': frame_count,
            'scene_filter': 'scene_D' if FILTER_SCENE_D else 'all',
        })

    rows.sort(key=lambda x: (x['task'], x['start_frame'], x['end_frame']))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'seg_id',
                'task',
                'instruction',
                'start_frame',
                'end_frame',
                'frame_count',
                'scene_filter',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved {len(rows)} rows to: {OUT_CSV}')
    print('Counts by task:')
    counts = {}
    for r in rows:
        counts[r['task']] = counts.get(r['task'], 0) + 1
    for task, count in sorted(counts.items()):
        print(f'  {task}: {count}')


if __name__ == '__main__':
    main()
