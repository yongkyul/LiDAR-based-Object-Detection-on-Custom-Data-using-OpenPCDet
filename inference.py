import os
import numpy as np
import torch
import open3d as o3d
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.sample_file_list = [f for f in os.listdir(root_path) if f.endswith('.bin')]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        input_dict = {
            'points': self.get_lidar(index),
            'frame_id': index
        }
        return self.prepare_data(input_dict)

    def get_lidar(self, index):
        # Load the point cloud data
        lidar_file = os.path.join(self.root_path, self.sample_file_list[index])
        assert os.path.exists(lidar_file), f'File {lidar_file} not found'
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return points

def load_model(config_file, checkpoint_file):
    cfg_from_yaml_file(config_file, cfg)
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=CustomDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False))
    model.load_params_from_file(checkpoint_file, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return model

def visualize_point_cloud(points, pred_boxes):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * [0.5, 0.5, 0.5])

    # Create Open3D bounding boxes
    geometries = [pcd]
    for box in pred_boxes:
        center = box[:3]
        size = box[3:6]
        orientation = box[6]

        # Create bounding box
        bbox = o3d.geometry.OrientedBoundingBox(center, size, orientation)
        bbox.color = (1, 0, 0)
        geometries.append(bbox)

    # Visualize point cloud and bounding boxes
    o3d.visualization.draw_geometries(geometries)

def inference(model, data_loader):
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data_dict = data
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            points = data_dict['points'][:, 1:].cpu().numpy()
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            
            visualize_point_cloud(points, pred_boxes)

def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            data_dict[key] = torch.from_numpy(val).float().cuda()

if __name__ == "__main__":
    config_file = './tools/cfgs/custom_models/pv_rcnn.yaml'
    checkpoint_file = './output/custom_models/pv_rcnn/default/ckpt/checkpoint_epoch_80.pth'
    data_path = './inference_data/'
    save_path = './inference_output'
    
    model = load_model(config_file, checkpoint_file)
    
    dataset = CustomDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False, root_path=data_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    inference(model, data_loader)
