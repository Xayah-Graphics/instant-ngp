#!/usr/bin/env python3

import argparse
import dataclasses
import json
import math
import pathlib

try:
    import PIL.Image
    import safetensors
    import torch
except ModuleNotFoundError as error:
    raise RuntimeError("example.py requires torch, safetensors, and Pillow.") from error


@dataclasses.dataclass(frozen=True)
class InstantNGPArchitecture:
    grid_n_levels: int = 8
    grid_features_per_level: int = 4
    grid_base_resolution: int = 16
    grid_log2_hashmap_size: int = 19
    grid_per_level_scale: float = 2.0
    grid_log2_per_level_scale: float = 1.0
    mlp_width: int = 64
    density_hidden_layers: int = 1
    rgb_hidden_layers: int = 2
    density_output_width: int = 16
    direction_output_width: int = 16
    network_output_width: int = 16
    nerf_grid_size: int = 128
    nerf_steps: int = 1024
    nerf_min_optical_thickness: float = 0.01
    transmittance_epsilon: float = 1e-4
    scene_scale: float = 0.33
    scene_offset: float = 0.5

    @property
    def grid_output_width(self) -> int:
        return self.grid_n_levels * self.grid_features_per_level

    @property
    def rgb_input_width(self) -> int:
        return self.density_output_width + self.direction_output_width

    @property
    def nerf_grid_cells(self) -> int:
        return self.nerf_grid_size * self.nerf_grid_size * self.nerf_grid_size

    @property
    def min_cone_stepsize(self) -> float:
        return math.sqrt(3.0) / self.nerf_steps


ARCHITECTURE = InstantNGPArchitecture()
EXPECTED_METADATA = {
    "format": "instant-ngp-new.weights.v1",
    "grid_n_levels": "8",
    "grid_features_per_level": "4",
    "grid_base_resolution": "16",
    "grid_log2_hashmap_size": "19",
    "grid_per_level_scale": "2",
    "grid_log2_per_level_scale": "1",
    "mlp_width": "64",
    "density_hidden_layers": "1",
    "rgb_hidden_layers": "2",
    "density_output_width": "16",
    "direction_output_width": "16",
    "rgb_input_width": "32",
    "network_output_width": "16",
    "grid_offsets": "0,4096,36864,299008,823296,1347584,1871872,2396160,2920448",
    "density_param_count": "3072",
    "rgb_param_count": "7168",
    "mlp_param_count": "10240",
    "grid_param_count": "11681792",
    "total_param_count": "11692032",
}
EXPECTED_TENSORS = {
    "density_mlp.input.weight": (ARCHITECTURE.mlp_width, ARCHITECTURE.grid_output_width),
    "density_mlp.output.weight": (ARCHITECTURE.density_output_width, ARCHITECTURE.mlp_width),
    "rgb_mlp.input.weight": (ARCHITECTURE.mlp_width, ARCHITECTURE.rgb_input_width),
    "rgb_mlp.hidden.weight": (ARCHITECTURE.mlp_width, ARCHITECTURE.mlp_width),
    "rgb_mlp.output.weight": (ARCHITECTURE.network_output_width, ARCHITECTURE.mlp_width),
    "hash_grid.params": (2920448, ARCHITECTURE.grid_features_per_level),
}


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    camera: torch.Tensor
    width: int
    height: int
    focal_length: float


class HashGridEncoder(torch.nn.Module):
    def __init__(self, architecture: InstantNGPArchitecture, dtype: torch.dtype) -> None:
        super().__init__()
        self.architecture = architecture
        self.params = torch.nn.Parameter(torch.empty(EXPECTED_TENSORS["hash_grid.params"], dtype=dtype), requires_grad=False)
        self.grid_offsets = tuple(int(value) for value in EXPECTED_METADATA["grid_offsets"].split(","))
        self.register_buffer("corner_offsets", torch.tensor([[corner & 1, (corner >> 1) & 1, (corner >> 2) & 1] for corner in range(8)], dtype=torch.int64), persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(torch.float32)
        encoded_levels = []
        for level in range(self.architecture.grid_n_levels):
            level_offset = self.grid_offsets[level]
            next_level_offset = self.grid_offsets[level + 1]
            level_grid = self.params[level_offset:next_level_offset]
            hashmap_size = next_level_offset - level_offset
            scale = math.exp2(level * self.architecture.grid_log2_per_level_scale) * self.architecture.grid_base_resolution - 1.0
            resolution = math.ceil(scale) + 1
            position = positions * scale + 0.5
            position_floor = torch.floor(position)
            position_grid = position_floor.to(torch.int64)
            fraction = position - position_floor
            level_result = torch.zeros((positions.shape[0], self.architecture.grid_features_per_level), device=positions.device, dtype=self.params.dtype)

            for corner in range(8):
                high_x = (corner & 1) != 0
                high_y = (corner & 2) != 0
                high_z = (corner & 4) != 0
                weight_x = fraction[:, 0] if high_x else 1.0 - fraction[:, 0]
                weight_y = fraction[:, 1] if high_y else 1.0 - fraction[:, 1]
                weight_z = fraction[:, 2] if high_z else 1.0 - fraction[:, 2]
                corner_offset = self.corner_offsets[corner]
                grid_position = position_grid + corner_offset
                feature_index = self.grid_index(hashmap_size, resolution, grid_position[:, 0], grid_position[:, 1], grid_position[:, 2])
                weight = (weight_x * weight_y * weight_z).to(self.params.dtype).unsqueeze(1)
                level_result = level_result + weight * level_grid[feature_index]

            encoded_levels.append(level_result)

        return torch.cat(encoded_levels, dim=1).to(self.params.dtype)

    def grid_index(self, hashmap_size: int, resolution: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if resolution <= 0x659:
            stride = resolution * resolution * resolution
            index = x + y * resolution + z * resolution * resolution
        else:
            stride = 0xFFFFFFFF
            index = torch.zeros_like(x)

        if hashmap_size < stride:
            y_hash = torch.bitwise_and(y * 2654435761, 0xFFFFFFFF)
            z_hash = torch.bitwise_and(z * 805459861, 0xFFFFFFFF)
            index = torch.bitwise_xor(torch.bitwise_xor(x, y_hash), z_hash)

        return torch.remainder(index, hashmap_size).to(torch.int64)


class SphericalHarmonicsEncoder(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        directions = directions.to(torch.float32)
        x = directions[:, 0]
        y = directions[:, 1]
        z = directions[:, 2]
        xy = x * y
        xz = x * z
        yz = y * z
        x2 = x * x
        y2 = y * y
        z2 = z * z
        encoded = torch.stack(
            (
                torch.full_like(x, 0.28209479177387814),
                -0.48860251190291987 * y,
                0.48860251190291987 * z,
                -0.48860251190291987 * x,
                1.0925484305920792 * xy,
                -1.0925484305920792 * yz,
                0.94617469575755997 * z2 - 0.31539156525251999,
                -1.0925484305920792 * xz,
                0.54627421529603959 * x2 - 0.54627421529603959 * y2,
                0.59004358992664352 * y * (-3.0 * x2 + y2),
                2.8906114426405538 * xy * z,
                0.45704579946446572 * y * (1.0 - 5.0 * z2),
                0.3731763325901154 * z * (5.0 * z2 - 3.0),
                0.45704579946446572 * x * (1.0 - 5.0 * z2),
                1.4453057213202769 * z * (x2 - y2),
                0.59004358992664352 * x * (-x2 + 3.0 * y2),
            ),
            dim=1,
        )
        return encoded.to(self.dtype)


class InstantNGPNetwork(torch.nn.Module):
    def __init__(self, architecture: InstantNGPArchitecture, dtype: torch.dtype) -> None:
        super().__init__()
        self.architecture = architecture
        self.position_encoder = HashGridEncoder(architecture, dtype)
        self.direction_encoder = SphericalHarmonicsEncoder(dtype)
        self.density_mlp = torch.nn.Sequential(
            torch.nn.Linear(architecture.grid_output_width, architecture.mlp_width, bias=False, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(architecture.mlp_width, architecture.density_output_width, bias=False, dtype=dtype),
        )
        self.rgb_mlp = torch.nn.Sequential(
            torch.nn.Linear(architecture.rgb_input_width, architecture.mlp_width, bias=False, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(architecture.mlp_width, architecture.mlp_width, bias=False, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(architecture.mlp_width, architecture.network_output_width, bias=False, dtype=dtype),
        )

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        density_output = self.forward_density_raw(positions)
        encoded_directions = self.direction_encoder(directions)
        rgb_input = torch.cat((density_output, encoded_directions), dim=1).to(self.rgb_mlp[0].weight.dtype)
        rgb_output = self.rgb_mlp(rgb_input)
        return torch.sigmoid(rgb_output[:, 0:3].to(torch.float32)), torch.exp(density_output[:, 0].to(torch.float32))

    def forward_density(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.forward_density_raw(positions)[:, 0].to(torch.float32))

    def forward_density_raw(self, positions: torch.Tensor) -> torch.Tensor:
        encoded_positions = self.position_encoder(positions).to(self.density_mlp[0].weight.dtype)
        return self.density_mlp(encoded_positions)


class InstantNGPInference:
    def __init__(self, weights_path: pathlib.Path, device: str, dtype: str, sample_batch: int) -> None:
        if sample_batch <= 0:
            raise RuntimeError("sample_batch must be positive.")
        self.architecture = ARCHITECTURE
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        if self.device.type == "cpu" and dtype != "float32":
            raise RuntimeError("CPU inference requires --dtype float32.")
        if self.device.type not in {"cuda", "cpu"}:
            raise RuntimeError("device must be either 'cuda' or 'cpu'.")

        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.sample_batch = sample_batch
        torch.set_float32_matmul_precision("high")
        self.network = load_weights(weights_path, self.architecture, self.device, self.dtype)
        self.network.eval()
        self.forward_compiled = torch.compile(self.network, mode="max-autotune", fullgraph=True)
        self.density_compiled = torch.compile(self.network.forward_density, mode="max-autotune", fullgraph=True)
        self.occupancy_grid: torch.Tensor | None = None

    @torch.inference_mode()
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise RuntimeError("positions must have shape [N, 3].")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise RuntimeError("directions must have shape [N, 3].")
        if positions.shape[0] != directions.shape[0]:
            raise RuntimeError("positions and directions must contain the same number of samples.")

        positions = positions.to(self.device, torch.float32)
        directions = directions.to(self.device, torch.float32)
        if not torch.isfinite(positions).all() or not torch.isfinite(directions).all():
            raise RuntimeError("positions and directions must be finite.")
        if torch.any(positions < 0.0) or torch.any(positions > 1.0):
            raise RuntimeError("positions must be inside the unit AABB.")
        if positions.shape[0] == 0:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32), torch.empty((0,), device=self.device, dtype=torch.float32)

        rgb_chunks = []
        density_chunks = []
        for start in range(0, positions.shape[0], self.sample_batch):
            end = min(start + self.sample_batch, positions.shape[0])
            rgb, density = self.forward_compiled(positions[start:end], directions[start:end])
            rgb_chunks.append(rgb)
            density_chunks.append(density)

        return torch.cat(rgb_chunks, dim=0), torch.cat(density_chunks, dim=0)

    @torch.inference_mode()
    def rebuild_occupancy_grid(self) -> torch.Tensor:
        density_grid = torch.empty((self.architecture.nerf_grid_cells,), device=self.device, dtype=torch.float32)
        for start in range(0, self.architecture.nerf_grid_cells, self.sample_batch):
            end = min(start + self.sample_batch, self.architecture.nerf_grid_cells)
            index = torch.arange(start, end, device=self.device, dtype=torch.int64)
            z = torch.div(index, self.architecture.nerf_grid_size * self.architecture.nerf_grid_size, rounding_mode="floor")
            y = torch.div(index - z * self.architecture.nerf_grid_size * self.architecture.nerf_grid_size, self.architecture.nerf_grid_size, rounding_mode="floor")
            x = index - z * self.architecture.nerf_grid_size * self.architecture.nerf_grid_size - y * self.architecture.nerf_grid_size
            positions = torch.stack((x, y, z), dim=1).to(torch.float32).add_(0.5).mul_(1.0 / self.architecture.nerf_grid_size)
            density_grid[start:end] = self.density_compiled(positions) * self.architecture.min_cone_stepsize

        threshold = torch.minimum(torch.clamp_min(density_grid, 0.0).mean(), torch.tensor(self.architecture.nerf_min_optical_thickness, device=self.device, dtype=torch.float32))
        self.occupancy_grid = (density_grid > threshold).reshape(self.architecture.nerf_grid_size, self.architecture.nerf_grid_size, self.architecture.nerf_grid_size).contiguous()
        return self.occupancy_grid


def load_weights(weights_path: pathlib.Path, architecture: InstantNGPArchitecture, device: torch.device, dtype: torch.dtype) -> InstantNGPNetwork:
    with safetensors.safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata != EXPECTED_METADATA:
            raise RuntimeError("safetensors metadata does not match instant-ngp-new.weights.v1.")
        tensor_names = set(handle.keys())
        if tensor_names != set(EXPECTED_TENSORS):
            raise RuntimeError(f"safetensors tensors mismatch: {sorted(tensor_names)}")

        tensors = {name: handle.get_tensor(name).to(torch.float32).contiguous() for name in EXPECTED_TENSORS}

    for name, shape in EXPECTED_TENSORS.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}.")
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"{name} must be float32.")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains non-finite values.")

    network = InstantNGPNetwork(architecture, torch.float32)
    with torch.no_grad():
        network.position_encoder.params.copy_(tensors["hash_grid.params"])
        network.density_mlp[0].weight.copy_(tensors["density_mlp.input.weight"])
        network.density_mlp[2].weight.copy_(tensors["density_mlp.output.weight"])
        network.rgb_mlp[0].weight.copy_(tensors["rgb_mlp.input.weight"])
        network.rgb_mlp[2].weight.copy_(tensors["rgb_mlp.hidden.weight"])
        network.rgb_mlp[4].weight.copy_(tensors["rgb_mlp.output.weight"])

    return network.to(device=device, dtype=dtype)


def load_nerf_synthetic_camera(dataset_path: pathlib.Path, split: str, image_index: int, max_size: int | None) -> CameraFrame:
    split_json_path = dataset_path / f"transforms_{split}.json"
    with split_json_path.open("r", encoding="utf-8") as input_file:
        transforms = json.load(input_file)

    frames = transforms["frames"]
    if image_index < 0 or image_index >= len(frames):
        raise RuntimeError(f"image_index must be in [0, {len(frames) - 1}].")

    frame = frames[image_index]
    image_path = pathlib.Path(frame["file_path"])
    if image_path.suffix == "":
        image_path = image_path.with_suffix(".png")
    if not image_path.is_absolute():
        image_path = split_json_path.parent / image_path
    image_path = image_path.resolve()

    with PIL.Image.open(image_path) as image:
        width, height = image.size

    original_width = width
    if max_size is not None:
        if max_size <= 0:
            raise RuntimeError("max_size must be positive.")
        scale = min(1.0, max_size / max(width, height))
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    else:
        scale = 1.0

    transform_matrix = frame["transform_matrix"]
    camera = [[float(transform_matrix[row][column]) for row in range(4)] for column in range(4)]
    for i in range(4):
        camera[1][i] = -camera[1][i]
        camera[2][i] = -camera[2][i]

    camera[3][0] = camera[3][0] * ARCHITECTURE.scene_scale + ARCHITECTURE.scene_offset
    camera[3][1] = camera[3][1] * ARCHITECTURE.scene_scale + ARCHITECTURE.scene_offset
    camera[3][2] = camera[3][2] * ARCHITECTURE.scene_scale + ARCHITECTURE.scene_offset

    camera_row0 = [camera[0][0], camera[1][0], camera[2][0], camera[3][0]]
    camera_row1 = [camera[0][1], camera[1][1], camera[2][1], camera[3][1]]
    camera_row2 = [camera[0][2], camera[1][2], camera[2][2], camera[3][2]]
    ngp_camera = torch.tensor(
        [
            camera_row1[0], camera_row2[0], camera_row0[0],
            camera_row1[1], camera_row2[1], camera_row0[1],
            camera_row1[2], camera_row2[2], camera_row0[2],
            camera_row1[3], camera_row2[3], camera_row0[3],
        ],
        dtype=torch.float32,
    )
    focal_length = 0.5 * float(original_width) / math.tan(float(transforms["camera_angle_x"]) * 0.5)
    return CameraFrame(camera=ngp_camera, width=width, height=height, focal_length=focal_length * scale)


@torch.inference_mode()
def render_frame(model: InstantNGPInference, frame: CameraFrame, ray_batch: int, marcher: str) -> torch.Tensor:
    if ray_batch <= 0:
        raise RuntimeError("ray_batch must be positive.")
    if marcher not in {"occupancy", "dense"}:
        raise RuntimeError("marcher must be either 'occupancy' or 'dense'.")
    if marcher == "occupancy" and model.occupancy_grid is None:
        model.rebuild_occupancy_grid()

    camera = frame.camera.to(model.device)
    camera_x = camera[0:3]
    camera_y = camera[3:6]
    camera_z = camera[6:9]
    ray_origin = camera[9:12]
    pixel_count = frame.width * frame.height
    rendered = torch.zeros((pixel_count, 3), device=model.device, dtype=torch.float32)

    for start in range(0, pixel_count, ray_batch):
        end = min(start + ray_batch, pixel_count)
        ray_count = end - start
        global_pixel = torch.arange(start, end, device=model.device, dtype=torch.int64)
        pixel_x = torch.remainder(global_pixel, frame.width).to(torch.float32)
        pixel_y = torch.div(global_pixel, frame.width, rounding_mode="floor").to(torch.float32)
        u = (pixel_x + 0.5) / frame.width
        v = (pixel_y + 0.5) / frame.height
        ray_x = (u - 0.5) * frame.width / frame.focal_length
        ray_y = (v - 0.5) * frame.height / frame.focal_length
        ray_direction = camera_x.unsqueeze(0) * ray_x.unsqueeze(1) + camera_y.unsqueeze(0) * ray_y.unsqueeze(1) + camera_z.unsqueeze(0)
        direction_length = torch.linalg.vector_norm(ray_direction, dim=1)
        zero_direction = direction_length == 0.0
        if torch.any(zero_direction):
            ray_direction[zero_direction] = camera_z
            direction_length = torch.linalg.vector_norm(ray_direction, dim=1)
        if torch.any(direction_length == 0.0):
            raise RuntimeError("camera produced zero-length ray directions.")

        ray_direction = ray_direction / direction_length.unsqueeze(1)
        t = intersect_unit_aabb(ray_origin, ray_direction)
        active = torch.isfinite(t)
        t = torch.where(active, t + 0.5 * model.architecture.min_cone_stepsize, t)
        sample_counts = torch.zeros((ray_count,), device=model.device, dtype=torch.int64)
        sample_positions = []
        sample_directions = []
        sample_ray_indices = []
        sample_step_indices = []

        for _ in range(model.architecture.nerf_steps):
            if not torch.any(active):
                break

            position = ray_origin.unsqueeze(0) + ray_direction * t.unsqueeze(1)
            inside = active & torch.all((position >= 0.0) & (position <= 1.0), dim=1)
            if not torch.any(inside):
                break

            if marcher == "occupancy":
                voxel = torch.floor(position * model.architecture.nerf_grid_size).to(torch.int64)
                voxel_inside = inside & torch.all((voxel >= 0) & (voxel < model.architecture.nerf_grid_size), dim=1)
                occupied = torch.zeros_like(active)
                occupied[voxel_inside] = model.occupancy_grid[voxel[voxel_inside, 2], voxel[voxel_inside, 1], voxel[voxel_inside, 0]]
                sample_mask = inside & occupied
            else:
                sample_mask = inside

            if torch.any(sample_mask):
                ray_indices = torch.nonzero(sample_mask, as_tuple=False).flatten()
                sample_positions.append(position[sample_mask])
                sample_directions.append(ray_direction[sample_mask])
                sample_ray_indices.append(ray_indices)
                sample_step_indices.append(sample_counts[sample_mask])
                sample_counts[sample_mask] += 1

            active = inside
            if marcher == "occupancy":
                empty_mask = active & ~sample_mask
                if torch.any(sample_mask):
                    t[sample_mask] += model.architecture.min_cone_stepsize
                if torch.any(empty_mask):
                    t[empty_mask] = advance_to_next_density_voxel(model.architecture, t[empty_mask], position[empty_mask], ray_direction[empty_mask])
            else:
                t[active] += model.architecture.min_cone_stepsize

        if sample_positions:
            positions = torch.cat(sample_positions, dim=0)
            directions = torch.cat(sample_directions, dim=0)
            ray_indices = torch.cat(sample_ray_indices, dim=0)
            step_indices = torch.cat(sample_step_indices, dim=0)
            rgb, density = model.forward(positions, directions)
            alpha = 1.0 - torch.exp(-density * model.architecture.min_cone_stepsize)
            max_samples_per_ray = int(sample_counts.max().item())
            alpha_by_ray = torch.zeros((ray_count, max_samples_per_ray), device=model.device, dtype=torch.float32)
            rgb_by_ray = torch.zeros((ray_count, max_samples_per_ray, 3), device=model.device, dtype=torch.float32)
            alpha_by_ray[ray_indices, step_indices] = alpha
            rgb_by_ray[ray_indices, step_indices] = rgb
            survival = torch.cat((torch.ones((ray_count, 1), device=model.device, dtype=torch.float32), torch.cumprod(1.0 - alpha_by_ray[:, :-1], dim=1)), dim=1)
            weights = alpha_by_ray * survival * (survival >= model.architecture.transmittance_epsilon)
            rendered[start:end] = torch.clamp((weights.unsqueeze(2) * rgb_by_ray).sum(dim=1), 0.0, 1.0)

    return rendered.reshape(frame.height, frame.width, 3)


def intersect_unit_aabb(origin: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    inv_direction = 1.0 / direction
    t0 = -origin.unsqueeze(0) * inv_direction
    t1 = (1.0 - origin).unsqueeze(0) * inv_direction
    tmin = torch.maximum(torch.minimum(t0, t1).amax(dim=1), torch.zeros((direction.shape[0],), device=direction.device, dtype=torch.float32))
    tmax = torch.maximum(t0, t1).amin(dim=1)
    return torch.where(tmax >= tmin, tmin, torch.full_like(tmin, float("nan")))


def advance_to_next_density_voxel(architecture: InstantNGPArchitecture, t: torch.Tensor, position: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    scale = float(architecture.nerf_grid_size)
    p = (position - 0.5) * scale
    inv_direction = 1.0 / direction
    sign = torch.where(direction < 0.0, -torch.ones_like(direction), torch.ones_like(direction))
    target = (torch.floor(p + 0.5 + 0.5 * sign) - p) * inv_direction
    t_target = t + torch.clamp_min(torch.amin(target, dim=1) / scale, 0.0)
    return t + torch.ceil(torch.clamp_min((t_target - t) / architecture.min_cone_stepsize, 0.5)) * architecture.min_cone_stepsize


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch-native instant-ngp-new safetensors inference example.")
    parser.add_argument("--weights", required=True, type=pathlib.Path)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image-index", default=0, type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--max-size", default=None, type=int)
    parser.add_argument("--ray-batch", default=4096, type=int)
    parser.add_argument("--sample-batch", default=262144, type=int)
    parser.add_argument("--marcher", default="occupancy", choices=("occupancy", "dense"))
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise RuntimeError(f"dataset path '{args.dataset}' is not a directory.")
    if not args.weights.is_file():
        raise RuntimeError(f"weights path '{args.weights}' is not a file.")
    if args.output.parent != pathlib.Path("") and not args.output.parent.is_dir():
        raise RuntimeError(f"output parent directory '{args.output.parent}' does not exist.")
    if args.image_index < 0:
        raise RuntimeError("image-index must be non-negative.")

    torch.set_float32_matmul_precision("high")
    model = InstantNGPInference(args.weights, args.device, args.dtype, args.sample_batch)
    frame = load_nerf_synthetic_camera(args.dataset, args.split, args.image_index, args.max_size)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={model.device} dtype={args.dtype} compiled=True marcher={args.marcher}")
    image = render_frame(model, frame, args.ray_batch, args.marcher)
    image_bytes = torch.round(image.mul(255.0)).to(torch.uint8).cpu().contiguous().flatten().tolist()
    PIL.Image.frombytes("RGB", (frame.width, frame.height), bytes(image_bytes)).save(args.output)
    print(f"saved {args.output} ({frame.width}x{frame.height}, split={args.split}, image_index={args.image_index})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
