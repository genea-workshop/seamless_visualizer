# Seamless Interaction Visualizer

A lightweight renderer for SMPL-H motion sequences that generates video output from `.npz` motion data and audio. The repository is built around `run.py`, which loads SMPL-H pose parameters, applies a SMPL-X body model, and renders a final `.mp4` using `pyrender`.

Used for visualizing SMPL-H clips from the Meta [Seamless Interaction dataset](https://github.com/facebookresearch/seamless_interaction), including both single-person and dyadic rendering with audio mixing in the output video.

Based on the visualizer from Meta [Embody-3D](https://github.com/facebookresearch/embody-3d).

## Repository structure

- `seamless_visualizer/run.py` - main script and CLI entrypoint
- `seamless_visualizer/visualize.py` - rendering pipeline and `PyrenderRenderer`
- `seamless_visualizer/load_obj.py` - OBJ topology loader for SMPL meshes
- `seamless_visualizer/enums.py` - dataset feature keys and annotations
- `seamless_visualizer/smplh_files/` - SMPL topology and texture assets
- `seamless_visualizer/utils/` - helper assets

## Features

- Render SMPL-H pose sequences from `.npz` files
- Support monadic and multiperson rendering
- Includes texture selection for male/female characters

## Requirements

- Python 3.11 (recommended)
- `numpy`
- `torch`
- `torchaudio`
- `smplx`
- `ffmpeg-python`
- `pyrender`
- `trimesh`
- `Pillow`
- `opencv-python`
- `PyOpenGL`
- `tqdm`
- torchcodec
- System `ffmpeg` installed and available on `PATH`


## Installation

1. Create and activate a virtual environment:

```bash
conda create -n seamless_vis python=3.11 -y
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install numpy torch torchaudio smplx ffmpeg-python pyrender trimesh Pillow opencv-python PyOpenGL tqdm torchcodec
```

3. Make sure the `ffmpeg` executable is installed on your system:

```bash
ffmpeg -version
```

## Usage

Run the renderer from the repository root:

```bash
python seamless_visualizer/run.py \
  --motion_npz path_to_file_1.npz path_to_file_2.npz \
  --audio_files path_to_file_1.wav path_to_file_2.wav \
  --output_dir output_folder \
  --multiperson True \
  --starting_second 0 \
  --ending_second 20
```

For a single-person render, provide only one motion file and one optional audio file:

```bash
python seamless_visualizer/run.py \
  --motion_npz path_to_file_1.npz \
  --audio_files path_to_file_1.wav \
  --multiperson False \
  --starting_second 0 \
  --ending_second 20
```

### Important CLI arguments

- `--smplh_model_path` - path to the SMPL-X/SMPL-H model folder (`seamless_visualizer/smplh_files` by default)
- `--smplh_topology_path` - path to the SMPL UV topology OBJ file (`seamless_visualizer/smplh_files/smpl_uv.obj` by default)
- `--output_dir` - directory where rendered videos are saved
- `--device` - compute device, defaults to `cuda` if available else `cpu`
- `--audio_files` - list of one or two audio file paths
- `--motion_npz` - list of one or two SMPL-H `.npz` motion files
- `--multiperson` - `True` for two-person rendering, `False` for a single person
- `--gender` - gender values for each person: `male`, `female`, `neutral`, or `auto`
- `--starting_second`, `--ending_second` - clip time window in seconds
- `--frame_rate` - frame rate used while sampling the `.npz` motion data

## Input data format

The renderer expects `.npz` motion files containing SMPL-H parameters with keys such as:

- `smplh:body_pose`
- `smplh:global_orient`
- `smplh:left_hand_pose`
- `smplh:right_hand_pose`

The script automatically zeroes lower-body motion and corrects a neck artifact in the loaded pose data.

## Audio handling

- For two-person rendering, the code mixes the signals into stereo so that each speaker is placed in a different audio channel.
- If `--gender auto` is used, the script reads `seamless_visualizer/utils/speakers_gender.json` to infer speaker gender when audio data is available.

## Output

Rendered videos are saved into the directory specified by `--output_dir`. The default output path is `seamless_visualizer/output_folder`.

The script writes either:

- a final `*.mp4` with embedded audio, or
- a silent `*.mp4` if no audio is provided.

## Notes

- ⚠️ You must download the required files from the official SMPL website before running the code and place them under `smplh_files`.
- ⚠️ This repository is under development.
