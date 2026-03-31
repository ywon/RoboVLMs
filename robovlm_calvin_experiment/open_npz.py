'''
import numpy as np

def inspect_npz(path):
    with np.load(path, allow_pickle=True) as data:
        print(f"[FILE] {path}")
        print("=" * 60)
        print("keys:", data.files)
        print()

        for k in data.files:
            v = data[k]
            print(f"[{k}]")
            print("  shape:", getattr(v, "shape", None))
            print("  dtype:", getattr(v, "dtype", type(v)))
            if isinstance(v, np.ndarray) and v.ndim == 1:
                print("  first values:", v[:10])
            print()



from pprint import pprint

path = "/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D/training/lang_annotations/auto_lang_ann.npy"
obj = np.load(path, allow_pickle=True).item()

for top_k, top_v in obj.items():
    print(f"\n================ {top_k} ================")

    if isinstance(top_v, dict):
        for sub_k, sub_v in top_v.items():
            if sub_k == "emb":
                print(f"\n----- {sub_k} -----")
                print("shape:", sub_v.shape)
                continue

            print(f"\n----- {sub_k} -----")
            print("type:", type(sub_v))
            print("len :", len(sub_v))
            pprint(sub_v[:10])

#inspect_npz("/home/yewon/RoboVLMs/calvin/dataset/calvin_debug_dataset/training/episode_0358482.npz")


import numpy as np

path = "/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D/training/lang_annotations/auto_lang_ann.npy"
obj = np.load(path, allow_pickle=True).item()

tasks = obj["language"]["task"]
indx = obj["info"]["indx"]

TOTAL_FRAMES = 2406144  # scene_info 기준 마지막 episode index + 1 로 가정

print("len(tasks):", len(tasks))
print("len(indx):", len(indx))
print("same length:", len(tasks) == len(indx))

# 1) 기본 구간 이상 여부
invalid_ranges = []
out_of_bounds = []

for i, (start, end) in enumerate(indx):
    if start > end:
        invalid_ranges.append((i, start, end))
    if start < 0 or end >= TOTAL_FRAMES:
        out_of_bounds.append((i, start, end))

print("\n[Basic checks]")
print("invalid_ranges:", len(invalid_ranges))
print("out_of_bounds:", len(out_of_bounds))

if invalid_ranges[:10]:
    print("first invalid ranges:", invalid_ranges[:10])
if out_of_bounds[:10]:
    print("first out_of_bounds:", out_of_bounds[:10])

# 2) 정렬 후 overlap / gap 체크
sorted_segments = sorted(
    [(start, end, i, tasks[i]) for i, (start, end) in enumerate(indx)],
    key=lambda x: (x[0], x[1])
)

overlaps = []
gaps = []

prev_start, prev_end, prev_i, prev_task = sorted_segments[0]

for cur_start, cur_end, cur_i, cur_task in sorted_segments[1:]:
    if cur_start <= prev_end:
        overlaps.append({
            "prev_idx": prev_i,
            "prev_task": prev_task,
            "prev_range": (prev_start, prev_end),
            "cur_idx": cur_i,
            "cur_task": cur_task,
            "cur_range": (cur_start, cur_end),
        })
        # 더 긴 끝점 유지
        if cur_end > prev_end:
            prev_end = cur_end
    else:
        if cur_start > prev_end + 1:
            gaps.append((prev_end + 1, cur_start - 1))
        prev_start, prev_end, prev_i, prev_task = cur_start, cur_end, cur_i, cur_task

print("\n[Overlap / Gap checks]")
print("num_overlaps:", len(overlaps))
print("num_gaps:", len(gaps))

if overlaps[:5]:
    print("\nfirst 5 overlaps:")
    for x in overlaps[:5]:
        print(x)

if gaps[:5]:
    print("\nfirst 5 gaps:")
    for x in gaps[:5]:
        print(x)

# 3) 커버 프레임 수 계산
total_segment_frames = sum(end - start + 1 for start, end in indx)

# union 길이 계산
merged = []
for start, end, _, _ in sorted_segments:
    if not merged or start > merged[-1][1] + 1:
        merged.append([start, end])
    else:
        merged[-1][1] = max(merged[-1][1], end)

unique_covered_frames = sum(end - start + 1 for start, end in merged)
uncovered_frames = TOTAL_FRAMES - unique_covered_frames

print("\n[Coverage summary]")
print("total_segment_frames (with overlap counted multiple times):", total_segment_frames)
print("unique_covered_frames:", unique_covered_frames)
print("uncovered_frames:", uncovered_frames)
print("coverage_ratio:", unique_covered_frames / TOTAL_FRAMES)

# 4) optional: task별 segment 개수
from collections import Counter
task_counter = Counter(tasks)

print("\n[Top 10 tasks by segment count]")
for task, cnt in task_counter.most_common(36):
    print(task, cnt)
'''
import numpy as np
from collections import Counter

ANN_PATH = "/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D/training/lang_annotations/auto_lang_ann.npy"

TARGET_TASKS = {
    "lift_blue_block_table",
    "lift_pink_block_table",
    "lift_red_block_table",
    "place_in_drawer",
}

SCENE_D_START = 0
SCENE_D_END = 611098

obj = np.load(ANN_PATH, allow_pickle=True).item()
tasks = obj["language"]["task"]
indx = obj["info"]["indx"]

counter = Counter()

for task, (start, end) in zip(tasks, indx):
    if task not in TARGET_TASKS:
        continue

    # scene D 안에 완전히 포함되는 segment만 count
    if SCENE_D_START <= start and end <= SCENE_D_END:
        counter[task] += 1

print("Scene D task counts:")
for t in sorted(TARGET_TASKS):
    print(f"{t}: {counter[t]}")