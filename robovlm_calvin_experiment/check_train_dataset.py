'''
import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

# CALVIN env
from calvin_env.envs.play_table_env import get_env

TARGET_TASKS = {
    "lift_blue_block_table",
    "lift_pink_block_table",
    "lift_red_block_table",
    "place_in_drawer",
}

TRAIN_DIR = Path("/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D/training")
ANN_PATH = TRAIN_DIR / "lang_annotations" / "auto_lang_ann.npy"
SCENE_INFO_PATH = TRAIN_DIR / "scene_info.npy"
OUT_DIR = Path("/home/yewon/RoboVLMs/robovlm_calvin_experiment/train_test")

SHOW_GUI = False
FPS = 10
MAX_PER_TASK = 5   # 테스트용이면 5~20 추천, 전부 돌리려면 None
RESIZE_TO = None      # 예: (256, 256)

def load_npz(frame_idx: int):
    path = TRAIN_DIR / f"episode_{frame_idx:07d}.npz"
    return np.load(path, allow_pickle=True)

def get_scene_name(frame_idx: int, scene_info: dict) -> str:
    for scene_name, (start, end) in scene_info.items():
        if start <= frame_idx <= end:
            return scene_name
    return "unknown"

def get_rgb_static(obs):
    # raw env obs는 보통 obs["rgb_obs"]["rgb_static"] 형태
    if "rgb_obs" in obs and "rgb_static" in obs["rgb_obs"]:
        return obs["rgb_obs"]["rgb_static"]
    # 혹시 키 구조가 다른 경우 대비
    if "rgb_static" in obs:
        return obs["rgb_static"]
    raise KeyError(f"rgb_static not found in obs keys: {obs.keys()}")

def maybe_resize(img):
    if RESIZE_TO is None:
        return img
    from PIL import Image
    return np.array(Image.fromarray(img).resize(RESIZE_TO))

def collect_task_segments():
    ann = np.load(ANN_PATH, allow_pickle=True).item()
    tasks = ann["language"]["task"]
    anns = ann["language"]["ann"]
    indx = ann["info"]["indx"]

    selected = []
    for i, (task, inst, (start, end)) in enumerate(zip(tasks, anns, indx)):
        if task in TARGET_TASKS:
            selected.append({
                "seg_id": i,
                "task": task,
                "instruction": inst,
                "start": int(start),
                "end": int(end),
            })
    return selected

def group_by_task(items):
    grouped = {}
    for x in items:
        grouped.setdefault(x["task"], []).append(x)
    return grouped

def replay_segment_to_gif(env, seg, scene_info):
    start_idx = seg["start"]
    end_idx = seg["end"]

    with load_npz(start_idx) as start_ep:
        init_robot_obs = start_ep["robot_obs"]
        init_scene_obs = start_ep["scene_obs"]

    scene_name = get_scene_name(start_idx, scene_info)

    # reset to exact dataset state
    obs = env.reset(robot_obs=init_robot_obs, scene_obs=init_scene_obs)

    frames = []
    first_img = get_rgb_static(obs)
    frames.append(maybe_resize(first_img))

    # replay dataset actions
    # 보통 현재 프레임의 rel_actions를 다음 상태로 가는 액션으로 사용
    for frame_idx in range(start_idx, end_idx):
        with load_npz(frame_idx) as ep:
            action = ep["rel_actions"].astype(np.float32)
        obs, reward, done, info = env.step(action)
        img = get_rgb_static(obs)
        frames.append(maybe_resize(img))

    task_dir = OUT_DIR / seg["task"]
    task_dir.mkdir(parents=True, exist_ok=True)

    gif_name = (
        f"{scene_name}_seg{seg['seg_id']:05d}_"
        f"{start_idx:07d}_{end_idx:07d}.gif"
    )
    gif_path = task_dir / gif_name
    imageio.mimsave(gif_path, frames, fps=FPS)
    return gif_path

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scene_info = np.load(SCENE_INFO_PATH, allow_pickle=True).item()
    segments = collect_task_segments()
    grouped = group_by_task(segments)

    print("Selected segment counts:")
    for task, segs in grouped.items():
        print(f"  {task}: {len(segs)}")

    # env 생성
    # wrapper 없이 raw env를 직접 쓰는 편이 GIF 뽑기 쉬움
    env = get_env(TRAIN_DIR, show_gui=SHOW_GUI)

    try:
        for task, segs in grouped.items():
            if MAX_PER_TASK is not None:
                segs = segs[:MAX_PER_TASK]

            print(f"\n[Task] {task} | replay {len(segs)} segments")
            for seg in tqdm(segs):
                try:
                    gif_path = replay_segment_to_gif(env, seg, scene_info)
                except Exception as e:
                    print(f"Failed on seg_id={seg['seg_id']} "
                          f"({seg['start']}-{seg['end']}): {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''
import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

from calvin_env.envs.play_table_env import get_env

TRAIN_DIR = Path("/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D/validation")
ANN_PATH = TRAIN_DIR / "lang_annotations" / "auto_lang_ann.npy"
OUT_DIR = Path("/home/yewon/RoboVLMs/robovlm_calvin_experiment/validation_test_envD")

TARGET_TASKS = {
    "lift_blue_block_table",
    "lift_pink_block_table",
    "lift_red_block_table",
    "place_in_drawer",
}

SCENE_D_START = 0
SCENE_D_END = 611098

FPS = 10
SHOW_GUI = False
MAX_PER_TASK = None  
SAVE_VIEW = "rgb_static"   # 또는 "rgb_gripper"


def load_npz(frame_idx: int):
    path = TRAIN_DIR / f"episode_{frame_idx:07d}.npz"
    return np.load(path, allow_pickle=True)


def collect_scene_d_segments():
    ann = np.load(ANN_PATH, allow_pickle=True).item()
    tasks = ann["language"]["task"]
    anns = ann["language"]["ann"]
    indx = ann["info"]["indx"]

    selected = []
    for seg_id, (task, inst, (start, end)) in enumerate(zip(tasks, anns, indx)):
        if task not in TARGET_TASKS:
            continue
        if not (SCENE_D_START <= start and end <= SCENE_D_END):
            continue

        selected.append({
            "seg_id": seg_id,
            "task": task,
            "instruction": inst,
            "start": int(start),
            "end": int(end),
        })
    return selected


def group_by_task(items):
    grouped = {}
    for x in items:
        grouped.setdefault(x["task"], []).append(x)
    return grouped


def get_img_from_obs(obs, key="rgb_static"):
    if "rgb_obs" in obs:
        return obs["rgb_obs"][key]
    return obs[key]


def replay_segment_and_save_gif(env, seg):
    start_idx = seg["start"]
    end_idx = seg["end"]

    with load_npz(start_idx) as ep0:
        init_robot_obs = ep0["robot_obs"]
        init_scene_obs = ep0["scene_obs"]

    obs = env.reset(robot_obs=init_robot_obs, scene_obs=init_scene_obs)

    frames = [get_img_from_obs(obs, SAVE_VIEW)]

    # start_idx ~ end_idx-1의 action으로 replay
    for frame_idx in range(start_idx, end_idx):
        with load_npz(frame_idx) as ep:
            action = ep["rel_actions"].astype(np.float32)
        obs, _, _, _ = env.step(action)
        frames.append(get_img_from_obs(obs, SAVE_VIEW))

    task_dir = OUT_DIR / seg["task"]
    task_dir.mkdir(parents=True, exist_ok=True)

    gif_name = f"sceneD_seg{seg['seg_id']:05d}_{start_idx:07d}_{end_idx:07d}.gif"
    gif_path = task_dir / gif_name
    imageio.mimsave(gif_path, frames, fps=FPS)
    return gif_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segments = collect_scene_d_segments()
    grouped = group_by_task(segments)

    print("Scene D selected segments:")
    for task, segs in grouped.items():
        print(f"  {task}: {len(segs)}")

    env = get_env(TRAIN_DIR, show_gui=SHOW_GUI)

    try:
        for task, segs in grouped.items():
            if MAX_PER_TASK is not None:
                segs = segs[:MAX_PER_TASK]

            print(f"\n[Task] {task} | saving {len(segs)} gifs")
            for seg in tqdm(segs):
                try:
                    replay_segment_and_save_gif(env, seg)
                except Exception as e:
                    print(f"Failed seg_id={seg['seg_id']} ({seg['start']}-{seg['end']}): {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()