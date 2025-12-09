import torch
import torch.onnx
import os
from mmcv import Config
from mmdet3d.models import build_detector
from mmdet.apis import set_random_seed
import numpy as np

def create_dummy_inputs(batch_size=1, num_points=10000, num_views=6):
    """Create dummy inputs for BEVFusion model."""
    # Dummy point cloud data (x, y, z, intensity)
    points = torch.randn(batch_size, num_points, 4).float()

    # Dummy image data - 6 views, RGB channels, 900x1600 resolution
    img = torch.randn(batch_size, num_views, 3, 900, 1600).float()

    # Dummy image metadata
    img_metas = []
    for _ in range(batch_size):
        meta = {
            'lidar2img': [
                np.eye(4), np.eye(4), np.eye(4),
                np.eye(4), np.eye(4), np.eye(4)
            ],
            'box_mode_3d': 'LIDAR',
            'pts_filename': 'dummy.npz',
            'flip': False,
            'pc_range': [-50, -50, -5, 50, 50, 3],
            'scale_factor': [1.0, 1.0],
            'img_shape': [900, 1600],
            'pad_shape': [900, 1600],
            'img_crop_shape': [900, 1600],
            'img_flip': False,
            'img_scale_factor': [1.0, 1.0]
        }
        img_metas.append(meta)

    return points, img, img_metas

def export_bevfusion_to_onnx(config_path, output_path, pretrained=None):
    """Export BEVFusion model to ONNX format."""
    # Load configuration
    cfg = Config.fromfile(config_path)

    # Set random seed
    set_random_seed(42)

    # Build model
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # Load pretrained weights if provided
    if pretrained and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained weights from {pretrained}")

    # Set model to evaluation mode
    model.eval()

    # Create dummy inputs
    points, img, img_metas = create_dummy_inputs()

    # Move model and inputs to CPU (required for ONNX export)
    model = model.cpu()
    points = points.cpu()
    img = img.cpu()
    img_metas_cpu = []
    for meta in img_metas:
        meta_cpu = {}
        for key, value in meta.items():
            if isinstance(value, np.ndarray):
                meta_cpu[key] = value
            elif isinstance(value, list):
                meta_cpu[key] = [v for v in value]
            else:
                meta_cpu[key] = value
        img_metas_cpu.append(meta_cpu)

    # Export to ONNX
    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        (points, img, img_metas_cpu),
        output_path,
        input_names=['points', 'img', 'img_metas'],
        output_names=['detection_results'],
        dynamic_axes={
            'points': {0: 'batch_size', 1: 'num_points'},
            'img': {0: 'batch_size', 1: 'num_views'},
            'detection_results': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True,
        verbose=True
    )

    print(f"Model successfully exported to {output_path}")
    print("ONNX model inputs:")
    print(f"- points: (batch_size, num_points, 4)")
    print(f"- img: (batch_size, num_views, 3, 900, 1600)")
    print(f"- img_metas: list of dictionaries with camera parameters")
    print("ONNX model output:")
    print("- detection_results: list of dictionaries with bounding boxes and scores")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export BEVFusion model to ONNX')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    args = parser.parse_args()

    export_bevfusion_to_onnx(args.config, args.output, args.pretrained)