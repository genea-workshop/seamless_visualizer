"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
import os
import platform
from typing import List

import numpy as np
import pyrender
import torch
import torch.nn as nn
import trimesh
from pyrender.constants import RenderFlags
from load_obj import load_obj
import PIL.Image as pimg
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
from OpenGL.GL import glBindBuffer, glBufferSubData, GL_ARRAY_BUFFER

from tqdm import tqdm

if platform.system() == "Darwin":
    # macOS uses the native windowing system (Cocoa/OpenGL)
    # Removing the environment variable usually lets it default correctly
    if "PYOPENGL_PLATFORM" in os.environ:
        del os.environ["PYOPENGL_PLATFORM"]
else:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    mp.set_start_method("spawn", force=True)
trimesh.util.log.setLevel("ERROR")

# Per-worker state (lives for the lifetime of the pool worker process)
_fast_worker = {}

def _init_fast_worker(renderer_args, faces, uvs_unrolled, shm_name, B, H, W, T_total, num_people, mask_obj_path):
    """Pool initializer: Added mask loading for worker processes."""
    rend = PyrenderRenderer(**renderer_args)
    rend.faces = faces
    rend._uvs_unrolled = uvs_unrolled
    
    # Load the mask in the worker context
    rend._load_mask(mask_obj_path)

    if num_people == 2:
        bg_img = cv2.imread("utils/white_bg2.png")
    else:
        bg_img = cv2.imread("utils/white_bg.png")
    bg_img = cv2.resize(bg_img, (W, H))
    bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

    shm = shared_memory.SharedMemory(name=shm_name)
    result = np.ndarray((B, T_total, 3, H, W), dtype=np.uint8, buffer=shm.buf)

    _fast_worker['rend'] = rend
    _fast_worker['bg'] = bg
    _fast_worker['shm'] = shm
    _fast_worker['result'] = result
    _fast_worker['mesh_nodes'] = None
    _fast_worker['mask_nodes'] = [] # Track masks per worker

def _render_chunk_fast(args):
    geom_chunk, textures, t_start = args
    rend    = _fast_worker['rend']
    bg      = _fast_worker['bg']
    result  = _fast_worker['result']

    n_persons = len(geom_chunk)
    mesh_nodes = _fast_worker['mesh_nodes']
    if mesh_nodes is None or len(mesh_nodes) != n_persons:
        mesh_nodes = [None] * n_persons

    B_local = geom_chunk[0].shape[0]
    T_chunk = geom_chunk[0].shape[1]

    for b in range(B_local):
        for t in range(T_chunk):
            for i, gc in enumerate(geom_chunk):
                # Update body geometry
                mesh_nodes[i] = rend._update_mesh_geometry(
                    gc[b, t], mesh_nodes[i], texture_path=textures[i]
                )
                # Update mask geometry for this person
                rend._update_mask_node_multi(gc[b, t], person_index=i)

            rgb, depth = rend.renderer.render(rend.scene, flags=rend.flags)
            mask = (depth > 0)[:, :, None]
            result[b, t_start + t] = np.where(mask, rgb, bg).transpose(2, 0, 1)

    _fast_worker['mesh_nodes'] = mesh_nodes
    return B_local * T_chunk

def split_chunks(geometry_list, num_workers):
    T = geometry_list[0].shape[1]
    chunk_size = T // num_workers

    chunks = []
    indices = []

    for i in range(num_workers):
        start = i * chunk_size
        end = T if i == num_workers - 1 else (i + 1) * chunk_size

        chunk = [g[:, start:end] for g in geometry_list]
        chunks.append(chunk)
        indices.append((start, end))

    return chunks, indices


class PyrenderRenderer(nn.Module):
    def __init__(
        self,
        topology_path: str,
        rendering_height: int = 512*2,
        rendering_width: int = 768*2*2,
        aspect_ratio: float = 3,
        camera_dist: int = -30,
        campos: torch.Tensor = None,
        magnification_factor: float = 1.0,  # 1000.0,
    ):
        super().__init__()
        topology_dict = load_obj(topology_path)
        self.aspect_ratio = aspect_ratio

        self.vi = topology_dict["vi"] # Vertex indices per face
        vt = topology_dict["vt"]  # Vertex topology
        vti = topology_dict["vti"] # Vertex topology indices

        self.remap_v_idx = []
        self.remap_vt = []
        self.remap_faces = []
        self.topology_path = topology_path

        # OBJ assigns different UV values to one vertex, but for pyrender/trimesh we need a 1 to 1 relationship
        idx = 0
        for face, face_uv in zip(self.vi, vti):
            face_indices = []
            for v_idx, vt_idx in zip(face, face_uv):
                self.remap_v_idx.append(v_idx)
                self.remap_vt.append(vt[vt_idx])
                face_indices.append(idx)
                idx += 1
            self.remap_faces.append(face_indices)

        self.remap_v_idx = np.array(self.remap_v_idx)
        self.uvs = np.array(self.remap_vt)
        self.faces = np.array(self.remap_faces)

        # Original topology for smooth normal computation.
        # run.py applies renderer.faces = renderer.faces[:, [0,2,1]] after construction,
        # so we pre-flip vi_np to match the winding that will be used at render time.
        self.vi_np = np.array([[f[0], f[1], f[2]] for f in self.vi], dtype=np.int32)  # [F, 3]
        self.vi_flipped = self.vi_np[:, [0, 2, 1]]  # [F, 3] — matches winding after run.py flip

        # Reverse map: orig_vertex_idx → one representative remap_idx
        # (used to reconstruct the 6890-vertex array from the unrolled one)
        n_orig = int(self.remap_v_idx.max()) + 1
        orig_to_remap = np.zeros(n_orig, dtype=np.int64)
        orig_to_remap[self.remap_v_idx] = np.arange(len(self.remap_v_idx), dtype=np.int64)
        self.orig_to_remap = orig_to_remap

        self.base_trimesh = trimesh.Trimesh(
            vertices=np.zeros((len(self.remap_v_idx), 3), dtype=np.float32),
            faces=self.faces,
            process=False
        )

        vertex_faces = defaultdict(list)
        for f_idx, face in enumerate(self.vi):
            for v in face:
                vertex_faces[v].append(f_idx)

        self.vertex_faces = vertex_faces

        self.texture_cache = {}
        self._bg = None

        self.mesh_nodes = []

        if campos is None:
            self.at = (0.0, -1.0, 0.0)
            self.azimuth = 0
            self.dist = camera_dist # We want to see the person closer
            self.elevation = -5.0
            self.light_loc = (0.0, 1.0, 15.0)
        else:
            raise NotImplementedError("configurable campos is not implemented yet")
        self.rendering_height = rendering_height * 2
        self.rendering_width = rendering_width * 2
        self.magnification_factor = magnification_factor
        self._mask_node = None
        self._setup_renderer()
        self._load_mask('utils/mask_triangulated.obj')
        
        # self._setup_floor() ## We don't need a floor because we are only generating the upper body and the camera is too close
    
    def _load_mask(self, mask_obj_path: str):
        """Load mask OBJ and extract material properties from the associated .mtl."""
        # trimesh will look for the .mtl file referenced inside the .obj automatically
        mesh = trimesh.load(mask_obj_path, force='mesh', process=False)
        mesh.vertices -= mesh.vertices.mean(axis=0)

        # Orientation fix (90° rotation)
        angle = np.deg2rad(90)
        Rx = np.array([
            [1,           0,            0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)],
        ], dtype=np.float32)
        
        mesh.vertices = mesh.vertices @ Rx.T
        self._mask_trimesh = mesh

        # --- EXTRACT MATERIAL FROM .MTL ---
        # Default fallback
        base_color = [1.0, 1.0, 1.0, 1.0] 
        texture = None

        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if hasattr(material, 'diffuse'):
                color = np.array(material.diffuse, dtype=np.float32)
                # Force normalization to 0.0 - 1.0
                if color.max() > 1.0:
                    color /= 255.0
                base_color = color.tolist()
                
                if len(base_color) == 3:
                    base_color.append(1.0) 

                self._mask_material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    emissiveFactor=[0.3, 0.3, 0.3],
                    metallicFactor=0.0,   # Set to 0 to avoid metallic reflections
                    roughnessFactor=1.0,    # Set to 1 to make it matte/non-shiny
                    alphaMode='OPAQUE'
                )

        # Create the pyrender material
        if texture:
            self._mask_material = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=texture,
                emissiveFactor=[0.3, 0.3, 0.3],
                metallicFactor=0.0,
                roughnessFactor=1.0,
                alphaMode='OPAQUE'
            )
        else:
            self._mask_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=base_color,
                emissiveFactor=[0.3, 0.3, 0.3],
                metallicFactor=0.0,
                roughnessFactor=1.0,
                alphaMode='OPAQUE'
            )


    def _compute_vertex_normals_torch(self, v: torch.Tensor):
        """Vectorized normal computation using PyTorch on GPU/CPU."""
        # Ensure faces are a torch tensor (do this once in __init__ for best perf)
        if not hasattr(self, 'face_tensor'):
            self.face_tensor = torch.from_numpy(self.vi.astype(np.int64)).to(v.device)

        # Get vertices of each face
        v_faces = v[self.face_tensor] # [F, 3, 3]

        # Compute face normals
        # (v1 - v0) x (v2 - v0)
        face_normals = torch.cross(
            v_faces[:, 1] - v_faces[:, 0],
            v_faces[:, 2] - v_faces[:, 0],
            dim=1
        )
        
        # Accumulate face normals to vertices
        v_normals = torch.zeros_like(v)

        v_normals.index_add_(0, self.face_tensor[:, 0], face_normals)
        v_normals.index_add_(0, self.face_tensor[:, 1], face_normals)
        v_normals.index_add_(0, self.face_tensor[:, 2], face_normals)

        # Normalize
        return torch.nn.functional.normalize(v_normals, p=2, dim=1)

    def _generate_checkerboard_geometry(
        self,
        length: float = 10,
        color0: List[float] = [0.4, 0.4, 0.4],
        color1: List[float] = [0.2, 0.2, 0.2],
        tile_width: float = 2,
        alpha: float = 1,
        up: str = "y",
    ):
        """helper function to generate a simple checkerboard geometry as the floor"""
        assert up == "y" or up == "z"
        color0 = np.array(color0 + [alpha])
        color1 = np.array(color1 + [alpha])
        radius = length / 2.0
        num_rows = num_cols = int(length / tile_width)
        vertices = []
        vert_colors = []
        faces = []
        face_colors = []
        for i in range(num_rows):
            for j in range(num_cols):
                u0, v0 = j * tile_width - radius, i * tile_width - radius
                us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
                vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
                zs = np.zeros(4)
                if up == "y":
                    cur_verts = np.stack([us, zs, vs], axis=-1)
                else:
                    cur_verts = np.stack([us, vs, zs], axis=-1)

                cur_faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int)
                cur_faces += 4 * (i * num_cols + j)
                use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
                cur_color = color0 if use_color0 else color1
                cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

                vertices.append(cur_verts)
                faces.append(cur_faces)
                vert_colors.append(cur_colors)
                face_colors.append(cur_colors)
        vertices = np.concatenate(vertices, axis=0).astype(np.float32)
        vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
        faces = np.concatenate(faces, axis=0).astype(np.float32)
        face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

        return vertices, faces, vert_colors, face_colors
    
    def _generate_solid_floor_geometry(
        self,
        length: float = 10,
        color: List[float] = [0.4, 0.4, 0.4],
        alpha: float = 1,
        up: str = "y",
    ):
        """helper function to generate a simple solid color geometry as the floor"""
        assert up == "y" or up == "z"

        full_color = np.array(color + [alpha], dtype=np.float32)
        radius = length / 2.0
        
        us = np.array([-radius, -radius, radius, radius])
        vs = np.array([-radius, radius, radius, -radius])
        zs = np.zeros(4)

        if up == "y":
            vertices = np.stack([us, zs, vs], axis=-1).astype(np.float32)
        else:
            vertices = np.stack([us, vs, zs], axis=-1).astype(np.float32)

        faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)

        vert_colors = np.tile(full_color, (4, 1))
        face_colors = np.tile(full_color, (4, 1))

        return vertices, faces, vert_colors, face_colors

    def _normalize(self, x):
        """Returns a normalized vector."""
        return x / torch.linalg.norm(x)

    def _viewmatrix(self, center, up, pos):
        """Returns a camera transformation matrix.

        Args:
            center: Point where the camera is looking.
            up: The upward direction of the camera.
            pos: The position of the camera.

        Returns:
            A camera transformation matrix.
        """
        lookat = center - pos
        vec2 = self._normalize(lookat)
        vec1_avg = self._normalize(up)
        vec0 = self._normalize(torch.cross(vec1_avg, vec2, dim=0))
        vec1 = self._normalize(torch.cross(vec2, vec0, dim=0))
        m = torch.stack([vec0, -vec1, -vec2, pos], 1)
        return m

    def _setup_renderer(self) -> None:
        """function to set up the scene with camera, lights, and renderer"""
        up_dir = 1
        self.scene = pyrender.Scene(ambient_light=np.array([0.8, 0.8, 0.8]), bg_color=(0.0, 0.0, 0.0, 0))
        camera = pyrender.PerspectiveCamera(yfov=(2 * np.pi / 180), aspectRatio=self.aspect_ratio)
        pos = torch.tensor([self.at[0], self.at[1] + self.elevation, self.at[2] + self.dist], dtype=torch.float)
        center = torch.tensor(self.at, dtype=torch.float)
        up = torch.tensor([0, up_dir, 0], dtype=torch.float)
        camRT = self._viewmatrix(center, up, pos)
        camRT = torch.cat([camRT, torch.tensor([[0, 0, 0, 1]])], dim=0)
        self.camera_node = self.scene.add(camera, pose=camRT, name="camera")

        # Enhanced lighting setup for better detail visibility with reduced harsh shadows
        # Main directional light (good detail visibility)
        main_light = pyrender.DirectionalLight(color=np.array([0.01, 0.01, 0.01]), intensity=1.0)
        main_light_pos = torch.tensor([0, -5e3, 5e3], dtype=torch.float)
        main_lightRT = self._viewmatrix(center, up, main_light_pos)
        main_lightRT = torch.cat([main_lightRT, torch.tensor([[0, 0, 0, 1]])], dim=0)
        self.main_light_node = self.scene.add(main_light, pose=main_lightRT, name="main_light")

        # Keep reference to main light for backwards compatibility
        self.light_node = self.main_light_node

        # Don't render the back faces
        self.flags = RenderFlags.SKIP_CULL_FACES

        # set up offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(self.rendering_width, self.rendering_height, point_size=6.0)

    def _setup_floor(self) -> None:
        """function to add the floor to the scene"""
        v, f, color_v, _ = self._generate_solid_floor_geometry()
        v[..., 1] += 0.01  # move the floor a bit down so we can see the feet better
        floor_tri = trimesh.creation.Trimesh(vertices=v, faces=f, face_colors=color_v, process=False)
        self.floor = pyrender.Mesh.from_trimesh(floor_tri, smooth=False)
        self.scene.add(self.floor)
    
    def forward(self, geometry_list, x_offset=[0, 0], textures=None, num_workers=4):
        num_people = len(geometry_list)
        B, T = geometry_list[0].shape[:2]
        H, W = self.rendering_height, self.rendering_width

        # Ensure UV unrolling is computed (uses current winding-flipped self.faces)
        if not hasattr(self, '_uvs_unrolled'):
            self._uvs_unrolled = self.uvs[self.faces].reshape(-1, 2).astype(np.float32)

        # Pre-process all geometry as one CPU numpy batch:
        #   - single GPU→CPU transfer per person instead of T small ones
        #   - transforms + vertex remap applied to [B, T, V, 3] at once
        geom_ready = []
        for i, g in enumerate(geometry_list):
            gn = np.array(g.detach().cpu(), dtype=np.float32)  # [B, T, V, 3]
            gn *= self.magnification_factor
            gn[..., 0] += x_offset[i]
            gn[..., 1] -= 0.45
            geom_ready.append(gn[:, :, self.remap_v_idx])  # [B, T, N_remap, 3]

        if num_workers == 1:
            return self._forward_single(geom_ready, textures, B, T, H, W, num_people)
        else:
            return self._forward_parallel(geom_ready, textures, B, T, H, W, num_people, num_workers)

    def _forward_single(self, geom_ready, textures, B, T, H, W, num_people):
        """Single-process rendering path."""
        if self._bg is None:
            if num_people == 2:
                bg_img = cv2.imread("utils/white_bg2.png")
            else:
                bg_img = cv2.imread("utils/white_bg.png")
            bg_img = cv2.resize(bg_img, (W, H))
            self._bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        result = np.empty((B, T, 3, H, W), dtype=np.uint8)
        mesh_nodes = [None] * len(geom_ready)

        with tqdm(total=B * T) as pbar:
            for b in range(B):
                for t in range(T):
                    for i, g_remap in enumerate(geom_ready):
                        mesh_nodes[i] = self._update_mesh_geometry(
                            g_remap[b, t], mesh_nodes[i], texture_path=textures[i]
                        )
                        
                        if hasattr(self, '_mask_trimesh'):
                            self._update_mask_node_multi(g_remap[b, t], person_index=i)

                    rgb, depth = self.renderer.render(self.scene, flags=self.flags)
                    mask = (depth > 0)[:, :, None]
                    result[b, t] = np.where(mask, rgb, self._bg).transpose(2, 0, 1)
                    pbar.update(1)
        return result

    def _forward_parallel(self, geom_ready, textures, B, T, H, W, num_people, num_workers):
        """Multi-process rendering path: splits frames across num_workers EGL contexts.

        Workers are initialized once (EGL context created once per worker), then fed
        many small tasks so tqdm shows smooth per-task progress.
        """
        renderer_args = dict(
            topology_path=self.topology_path,
            rendering_height=H // 2,
            rendering_width=W // 2,
            aspect_ratio=self.aspect_ratio,
            magnification_factor=self.magnification_factor,
        )

        shm = shared_memory.SharedMemory(create=True, size=B * T * 3 * H * W)
        result_shm = np.ndarray((B, T, 3, H, W), dtype=np.uint8, buffer=shm.buf)

        # Small task size (~8 frames) → many tasks → smooth tqdm updates.
        # Workers keep their EGL context and mesh_nodes across tasks (no re-init).
        task_size = max(1, min(8, T // num_workers))
        tasks = []
        t = 0
        while t < T:
            t_end = min(t + task_size, T)
            geom_chunk = [g[:, t:t_end] for g in geom_ready]
            tasks.append((geom_chunk, textures, t))
            t = t_end

        mask_path = 'utils/mask_triangulated.obj'

        initargs = (
            renderer_args,
            self.faces, self._uvs_unrolled,
            shm.name, B, H, W, T,
            num_people,
            mask_path # Pass the path here
        )
        
        with mp.Pool(
            processes=num_workers,
            initializer=_init_fast_worker,
            initargs=initargs,
        ) as pool:
            with tqdm(total=B * T, desc="Rendering") as pbar:
                for frames_done in pool.imap_unordered(_render_chunk_fast, tasks):
                    pbar.update(frames_done)

        result = result_shm.copy()
        shm.close()
        shm.unlink()
        return result
    
    def _update_primitive_buffer(self, primitive, positions, normals):
        """Update vertex positions and normals in the existing GPU buffer in-place.

        Avoids recreating the VAO/VBO and re-uploading the texture every frame.
        The buffer layout must match _add_to_context: [pos(3), norm(3), uv(2)] interleaved.
        """
        vertex_data = np.ascontiguousarray(
            np.hstack((positions, normals, self._uvs_unrolled)).flatten().astype(np.float32)
        )
        glBindBuffer(GL_ARRAY_BUFFER, primitive._buffers[0])
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def _update_mask_node_multi(self, v_remapped: np.ndarray, person_index: int, scale: float = 1.0):
        pose = self._get_face_pose(v_remapped)
        pose[:3, :3] *= scale

        if not hasattr(self, '_mask_nodes'):
            self._mask_nodes = []
        if not hasattr(self, '_mask_light_nodes'):
            self._mask_light_nodes = []
        
        while len(self._mask_nodes) <= person_index:
            self._mask_nodes.append(None)
        while len(self._mask_light_nodes) <= person_index:
            self._mask_light_nodes.append(None)

        # Reuse the mesh object, but remove the old NODE from the scene
        if not hasattr(self, '_mask_mesh'):
            material = getattr(self, '_mask_material', None)
            self._mask_mesh = pyrender.Mesh.from_trimesh(self._mask_trimesh, material=material, smooth=False)

        # Clean up old frame's nodes
        if self._mask_nodes[person_index] is not None:
            try:
                self.scene.remove_node(self._mask_nodes[person_index])
            except ValueError: pass
        
        if self._mask_light_nodes[person_index] is not None:
            try:
                self.scene.remove_node(self._mask_light_nodes[person_index])
            except ValueError: pass

        # Add the mask mesh
        self._mask_nodes[person_index] = self.scene.add(self._mask_mesh, pose=pose)

        # Add a directional light to light up the mask
        face_light = pyrender.DirectionalLight(color=np.ones(3), intensity=0.4)
        light_pose = pose.copy()
        self._mask_light_nodes[person_index] = self.scene.add(face_light, pose=light_pose)

    def _update_mesh_geometry(self, v_remapped: np.ndarray, mesh_node, texture_path: str = 'smplh_files/textures/smplx_texture_m_alb.png'):
        # Unroll vertices by face (winding-flipped faces)
        v_faces = v_remapped[self.faces]  # [F, 3, 3]
        positions = np.ascontiguousarray(v_faces.reshape(-1, 3).astype(np.float32))

        # Smooth vertex normals computed on the original 6890-vertex topology.
        # Each remapped vertex gets the smooth normal of its original counterpart → Gouraud shading
        v_orig = v_remapped[self.orig_to_remap]                     # [N_orig, 3]
        vf_orig = v_orig[self.vi_flipped]                            # [F, 3, 3]
        e1 = vf_orig[:, 1] - vf_orig[:, 0]
        e2 = vf_orig[:, 2] - vf_orig[:, 0]
        fn = np.cross(e1, e2)                                        # area-weighted face normals
        vn_orig = np.zeros_like(v_orig)
        np.add.at(vn_orig, self.vi_flipped[:, 0], fn)
        np.add.at(vn_orig, self.vi_flipped[:, 1], fn)
        np.add.at(vn_orig, self.vi_flipped[:, 2], fn)
        nl = np.linalg.norm(vn_orig, axis=1, keepdims=True)
        np.maximum(nl, 1e-8, out=nl)
        vn_orig /= nl                                                # [N_orig, 3] smooth normals

        # Map smooth normals onto the unrolled (face × 3) vertex layout
        vn_remap = vn_orig[self.remap_v_idx]                        # [N_remap, 3]
        normals = np.ascontiguousarray(
            vn_remap[self.faces].reshape(-1, 3).astype(np.float32)  # [F*3, 3]
        )

        # Cache material — reuse the SAME object so pyrender never re-uploads the texture
        if texture_path not in self.texture_cache:
            image = pimg.open(texture_path)
            tex = pyrender.Texture(source=image, source_channels='RGB')
            self.texture_cache[texture_path] = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=tex,
                metallicFactor=0,
                roughnessFactor=1.0
            )
        material = self.texture_cache[texture_path]

        if mesh_node is None:
            # First frame: build Primitive directly (no trimesh) and add to scene
            primitive = pyrender.Primitive(
                positions=positions,
                normals=normals,
                texcoord_0=self._uvs_unrolled,
                material=material,
            )
            return self.scene.add(pyrender.Mesh(primitives=[primitive]))
        else:
            # Subsequent frames: patch GPU buffer in-place — zero VAO/texture overhead
            self._update_primitive_buffer(
                mesh_node.mesh.primitives[0], positions, normals
            )
            return mesh_node
        
    def _update_mask_node(self, v_remapped: np.ndarray, scale: float = 1.0):
        """Add or update the mask mesh node in the scene."""
        pose = self._get_face_pose(v_remapped)
        pose[:3, :3] *= scale

        # Build the pyrender Mesh once and reuse it
        if not hasattr(self, '_mask_mesh'):
            self._mask_mesh = pyrender.Mesh.from_trimesh(self._mask_trimesh, smooth=False)
        
        if not hasattr(self, '_mask_light'):
            light = pyrender.PointLight(
                color=np.ones(3),
                intensity=0.04
            )
            self._mask_light = self.scene.add(light, pose=np.eye(4))
        
        # Offset the light slightly in front of the mask
        light_pose = pose.copy()
        light_pose[:3, 3] += pose[:3, 2] * 0.2

        self.scene.set_pose(self._mask_light, pose=light_pose)

        # remove and re-add each frame
        if self._mask_node is not None:
            self.scene.remove_node(self._mask_node)

        self._mask_node = self.scene.add(self._mask_mesh, pose=pose)

    def _get_face_pose(self, v_remapped: np.ndarray) -> np.ndarray:
        def get_v(orig_idx):
            return v_remapped[self.orig_to_remap[orig_idx]].copy()

        # Landmarks
        upper_nose_tip  = get_v(330)
        left_eye  = get_v(2800)
        right_eye = get_v(6260)
        lower_nose_tip  = get_v(332)
        
        # Vector from right eye to left eye
        right_vec = left_eye - right_eye
        right_vec /= (np.linalg.norm(right_vec) + 1e-8)

        # Vector in the upward direction
        approx_up = upper_nose_tip - lower_nose_tip
        approx_up /= (np.linalg.norm(approx_up) + 1e-8)

        # Forward Vector (Depth/Normal of the face)
        # This is perpendicular to the eye-line and the forehead-nose line, ensuring the mask is parallel to the 'plane' of the face
        forward = np.cross(right_vec, approx_up)
        forward /= (np.linalg.norm(forward) + 1e-8)

        # Corrected Up Vector
        # Re-calculate Up to ensure strict orthogonality (90 degrees to Forward and Right)
        # This removes the 'tilt' caused by the nose being further forward than the forehead
        up_vec = np.cross(forward, right_vec)
        up_vec /= (np.linalg.norm(up_vec) + 1e-8)

        # Center the mask on the bridge of the nose (between eyes) 
        # and push it slightly forward
        face_center = (left_eye + right_eye) / 2.0
        face_center += forward * 0.02

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right_vec
        pose[:3, 1] = up_vec
        pose[:3, 2] = forward
        pose[:3, 3] = face_center
        return pose