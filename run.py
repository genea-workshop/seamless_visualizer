"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

import subprocess

import ffmpeg
import smplx
import torch
import torchaudio
from enums import FeatName
from visualize import PyrenderRenderer

import numpy as np

def get_texture(gender, num_person):
    # Depending if it is person 1 or 2, we use a different T-shirt color for helping with differentiating people.
    if gender == "male":
        if num_person == 0:
            return "smplh_files/textures/smplx_texture_m_alb.png"
        else:
            return "smplh_files/textures/smplx_texture_m_alb_v2.png"
    else:
        if num_person == 0:
            return "smplh_files/textures/smplx_texture_f_alb.png"
        else:
            return "smplh_files/textures/smplx_texture_f_alb_v2.png"


def get_wav_duration(file_path):
    info = torchaudio.info(file_path)
    return int(info.num_frames / info.sample_rate)

def load_smplh(args, npz_file, seq_len=None):
    """
    Load SMPL-H poses stored in a .npz file.
    
    Args:
        npz_file (str): Path to the .npz file containing SMPL-H parameters.
        smplh_model_path (str): Folder containing SMPL-X model files.
        gender (str): 'male', 'female', or 'neutral'.
    
    Returns:
        dict: Contains 'vertices', 'joints', 'global_orient', 'body_pose', 
              'left_hand_pose', 'right_hand_pose', 'translation' as numpy arrays.
    """
    
    # Load SMPL-H data
    data = np.load(npz_file)

    start_sample = int(args.starting_second * args.frame_rate)
    end_sample = int(args.ending_second * args.frame_rate)

    if seq_len == None:
        seq_len = data['smplh:body_pose'].shape[0]
    body_pose = data['smplh:body_pose'][start_sample:end_sample]
    global_orient = data['smplh:global_orient'][start_sample:end_sample]
    left_hand_pose = data['smplh:left_hand_pose'][start_sample:end_sample]
    right_hand_pose = data['smplh:right_hand_pose'][start_sample:end_sample]

    # Zero-out the lower body motion, since it is not part of the seamless dataset
    lower_body_joints = [0, 1, 2, 3, 4, 6, 7, 9, 10]
    for j in lower_body_joints:
        body_pose[:, j] = 0

    # Correct neck artifact in the dataset
    mean_neck_j11 = np.mean(body_pose[:, 11], axis=0)
    body_pose[:, 11] -= mean_neck_j11
    mean_neck_j14 = np.mean(body_pose[:, 14], axis=0)
    body_pose[:, 14] -= mean_neck_j14

    # Convert to torch tensors
    body_pose_x = torch.tensor(body_pose, dtype=torch.float32)
    left_hand_x = torch.tensor(left_hand_pose, dtype=torch.float32)
    right_hand_x = torch.tensor(right_hand_pose, dtype=torch.float32)
    global_orient_x = torch.tensor(global_orient, dtype=torch.float32)
    # Set the orientation of the hips constant to avoid moving the legs entirely.
    global_orient_x = torch.tensor([torch.pi, 0, 0]).expand(global_orient_x.shape[0], -1)


    smplh_mesh = {
        'smplh_mesh_global_orient': global_orient_x.unsqueeze(0),
        'smplh_mesh_body_pose': body_pose_x.unsqueeze(0),
        'smplh_mesh_left_hand_pose': left_hand_x.unsqueeze(0),
        'smplh_mesh_right_hand_pose': right_hand_x.unsqueeze(0),
        'seq_name': Path(npz_file).stem,
        'starting_second': str(args.starting_second)
    }

    # Return everything as numpy arrays
    return smplh_mesh


def compute_lbs_from_batch_model(device_model, smplh_model, batch, smplh_keys):
    smplh_input = {}
    
    for key in batch.keys():
        if key in smplh_keys:
            clean_key = key.replace("smplh_mesh_", "").replace("smplh:", "")
            val = batch[key][0].to(dtype=torch.float32, device=device_model)
            
            # 1. Flatten pose tensors
            if clean_key in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']:
                val = val.reshape(val.shape[0], -1) 
            
            smplh_input[clean_key] = val

    # Now the forward pass will have matching dimensions
    output = smplh_model(**smplh_input)
    return output.vertices[None, ...]

def normalize_rms(audio_tensor, target_level=0.3):
    rms = audio_tensor.pow(2).mean().sqrt()
    if rms > 0:
        return audio_tensor * (target_level / rms)
    return audio_tensor

def load_and_visualize_data_model(args, gender_json, renderer=None):
    args.starting_second = math.floor(args.starting_second)
    args.ending_second = math.ceil(args.ending_second)
    
    seq_len = (args.ending_second - args.starting_second) * args.frame_rate
    x_offset = [0, 0]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ### set up the dataset
    smplh_keys = [
        FeatName.BODY.value,
        FeatName.ROT.value,
        FeatName.TRANS.value,
        FeatName.SHAPE.value,
        FeatName.LEFT_HAND.value,
        FeatName.RIGHT_HAND.value,
    ]

    batch = load_smplh(args, args.motion_npz[0], seq_len)

    rendering_height = 512

    if not args.multiperson:
        rendering_width = 640
        camera_dist = -40
        aspect_ratio = 1.25
    else:
        rendering_width = 768*2
        camera_dist = -30
        aspect_ratio = 3

    if renderer is None:
        t0 = time.time()
        renderer = PyrenderRenderer(topology_path=args.smplh_topology_path, rendering_height=rendering_height, rendering_width=rendering_width, aspect_ratio=aspect_ratio, camera_dist=camera_dist).to(args.device)
        renderer.faces = renderer.faces[:, [0, 2, 1]]

    people = [{
        "batch": batch,
        "gender": args.gender[0],
        "audio": args.audio_files[0] if len(args.audio_files) > 0 else None
    }]

    if args.multiperson:
        x_offset = [0.65, -0.65]
        other_batch = load_smplh(args, args.motion_npz[1], seq_len)
        people.append({
            "batch": other_batch,
            "gender": args.gender[1],
            "audio": args.audio_files[1] if len(args.audio_files) > 1 else None
        })

    # Gender detection
    for p in people:
        if p["gender"] == "auto" and p["audio"] is not None:
            audio, sr = torchaudio.load(p["audio"])  # [channels, time], int
            p["waveform"] = audio
            p["sr"] = sr

            p_id = p['batch']['seq_name'].split("_")[-1]
            p["gender"] = gender_json.get(p_id, 'male')

    smpl_model_cache = {}

    def get_smpl_model(gender):
        if gender not in smpl_model_cache:
            ### set up the smplx function
            smpl_model_cache[gender] = smplx.create(
                args.smplh_model_path,
                model_type="smplh",
                gender=gender,
                flat_hand_mean=True,
                num_betas=10,
                use_pca=False,
                batch_size=seq_len
            ).to(args.device)
        return smpl_model_cache[gender]

    # Build meshes
    verts_list, textures = [], []

    for count, p in enumerate(people):
        model = get_smpl_model(p["gender"])
        verts = compute_lbs_from_batch_model(args.device, model, p["batch"], smplh_keys)

        verts_list.append(verts)
        textures.append(get_texture(p["gender"], count))

    t0 = time.time()

    nodes_to_remove = [node for node in renderer.scene.mesh_nodes]
    for node in nodes_to_remove:
        renderer.scene.remove_node(node)

    video_frames = renderer(verts_list, x_offset=x_offset, textures=textures)[0]
    # Transpose (T,3,H,W) → (T,H,W,3) and ensure C-contiguous uint8 in one allocation.
    # ascontiguousarray on a non-contiguous transpose avoids the separate tobytes copy.
    t0 = time.time()
    video_frames = np.ascontiguousarray(video_frames.transpose(0, 2, 3, 1), dtype=np.uint8)

    # Save the silent video
    base_name = f"{batch['seq_name']}--{batch['starting_second']}"
    temp_video_path = f"{output_dir}/{base_name}_temp.mp4"
    final_path = f"{output_dir}/{base_name}.mp4"

    _, H, W, _ = video_frames.shape

    def _encode(encoder, extra_flags=()):
        if encoder == "h264_nvenc":
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", "30",
                "-i", "pipe:",
                "-vcodec", encoder, *extra_flags, "-pix_fmt", "yuv420p",
                temp_video_path,
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = proc.communicate(memoryview(video_frames))
            if proc.returncode != 0:
                raise RuntimeError(err.decode(errors="replace"))
        else:
            # Write raw frames to a temp file so ffmpeg can read them without pipe
            frames_path = f"{temp_video_path}.raw"
            with open(frames_path, "wb") as f:
                f.write(memoryview(video_frames))
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", "30",
                "-i", frames_path,
                "-vcodec", encoder, *extra_flags, "-pix_fmt", "yuv420p",
                temp_video_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            os.remove(frames_path)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace"))

    t0 = time.time()
    for encoder, flags in [
        ("h264_nvenc",  ("-preset", "fast")),
        ("libx264",     ("-preset", "fast", "-crf", "18")),
        ("libopenh264", ()),
        ("h264_vaapi",  ()),
    ]:
        try:
            _encode(encoder, flags)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("No suitable H.264 encoder found. Tried: h264_nvenc, libx264, libopenh264, h264_vaapi")

    if len(args.audio_files) > 0:

        start_sample = int(args.starting_second * args.sample_rate)
        end_sample = int(args.ending_second * args.sample_rate)

        audio_segments = []

        # Arrange the audio so that we can hear from the left to the person that we see on the left (same with the person on the right)
        for p in people:
            if "waveform" in p:
                segment = p["waveform"][:, start_sample:end_sample]

                # Ensure correct sample rate
                if p["sr"] != 48000:
                    segment = torchaudio.functional.resample(segment, p["sr"], 48000) # Shouldn't be needed for the Seamless dataset, but I keep it just in case some subset needs it

                audio_segments.append(segment)

        # Merge audio
        if len(audio_segments) == 2:
            # Force mono and then assign L/R --> Shouldn't be needed for the Seamless dataset, but I keep it just in case some subset needs it
            a1 = audio_segments[0].mean(dim=0, keepdim=True)
            a2 = audio_segments[1].mean(dim=0, keepdim=True)

            normalized_a1 = normalize_rms(a1, target_level=0.3)
            normalized_a2 = normalize_rms(a2, target_level=0.3)

            final_audio = torch.cat([normalized_a2, normalized_a1], dim=0)  # [2, T] stereo

        # If we only have one person, the audio goes to the two ears
        else:
            final_audio = audio_segments[0]

        temp_audio_path = f"{output_dir}/{base_name}_temp.wav"
        torchaudio.save(temp_audio_path, final_audio, 48000)

        video_input = ffmpeg.input(temp_video_path)
        audio_input = ffmpeg.input(temp_audio_path)

        (
            ffmpeg
            .output(
                video_input,
                audio_input,
                final_path,
                vcodec="copy",
                acodec="aac"
            )
            .global_args("-hide_banner", "-loglevel", "error", "-shortest")
            .run(overwrite_output=True)
        )

        # Cleanup
        os.remove(temp_video_path)
        os.remove(temp_audio_path)

        print(f"Saved final video with audio: {final_path}")

    else:
        os.rename(temp_video_path, final_path)
        print(f"Saved video (no audio found): {final_path}")
    return video_frames

def _make_renderer(args):
    rendering_height = 512
    if not args.multiperson:
        rendering_width = 640
        camera_dist = -40
        aspect_ratio = 1.25
    else:
        rendering_width = 768 * 2
        camera_dist = -30
        aspect_ratio = 3
    t0 = time.time()
    renderer = PyrenderRenderer(
        topology_path=args.smplh_topology_path,
        rendering_height=rendering_height,
        rendering_width=rendering_width,
        aspect_ratio=aspect_ratio,
        camera_dist=camera_dist,
    ).to(args.device)
    renderer.faces = renderer.faces[:, [0, 2, 1]]
    return renderer

def render_one_clip(args, gender_json):
    start_time = time.time()
    renderer = _make_renderer(args)
    load_and_visualize_data_model(args, gender_json, renderer=renderer)
    end_time = time.time()    # Record the end
    duration = end_time - start_time

    print(f"\nTotal execution time: {duration:.2f} seconds")


if __name__ == "__main__":

    motion_npz_list = [
        'path_to_file_1.npz',
        'path_to_file_2.npz' # Not necessary if only rendering monadic motion
    ]

    audio_files = [
        'path_to_file_1.wav',
        'path_to_file_2.wav' # Not necessary if only rendering monadic motion
    ]

    genders = ["auto", "auto"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--smplh_model_path", type=str, required=False, default='smplh_files', help="path to the smplx model")
    parser.add_argument("--smplh_topology_path", type=str, required=False, default='smplh_files/smpl_uv.obj', help="path to the smplx obj topology")
    parser.add_argument("--output_dir", type=str, required=False, default="output_folder", help="output directory to save the rendered videos")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() and torch.version.cuda is not None else "cpu", help="device to run the smplx model on")
    parser.add_argument("--audio_files", type=str, nargs='+', default=audio_files, help="Path to the audio file")
    parser.add_argument("--motion_npz", type=str, nargs='+', default=motion_npz_list)
    parser.add_argument("--multiperson", type=bool, default=True)
    parser.add_argument("--gender", type=str, default=genders)

    parser.add_argument("--starting_second", type=float, default=0)
    parser.add_argument("--ending_second", type=float, default=20)
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=48000)
    args = parser.parse_args()

    with open('utils/speakers_gender.json', 'r') as f:
        gender_json = json.load(f)

    render_one_clip(args, gender_json)