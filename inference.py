# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='panelnet', help='Dataset: sunrgbd or scannet or panelnet [default: panelnet]')
parser.add_argument('--checkpoint_dir',  help='checkpoint directory of the thrained weights.')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--use_height', type=bool, default=False, help='Use height? [default: False]')
parser.add_argument('--use_color', type=bool, default=False, help='Use height? [default: False]')
parser.add_argument('--pc_path', type=str, default='generate', help='Pointcloud absolute path. [generate] for random generation')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--rbw', action='store_true', help='Read mesh for RoboWeldAR pipeline and preprocess')



FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    if FLAGS.use_height:
        floor_height = np.percentile(point_cloud[:,2],0.99)
        print("floor_height", floor_height)
        height = point_cloud[:,2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def preprocess_rbw_mesh(mesh, filter_radius=0.7, negative_filter=-0.05):
    # ROBOWELDAR 3D RECONSTRUCTION SCAN PRE PROCESS
    # mesh.scale(3,mesh.get_center())
    point_cloud = mesh.sample_points_uniformly(number_of_points=int(FLAGS.num_point), use_triangle_normal=True)
    point_cloud = np.asarray(point_cloud.points)
    # crop in a semisphere
    point_cloud = point_cloud[np.linalg.norm(point_cloud, axis=1) < filter_radius]  # points that have euclidian distance from 0 < filter_radius
    point_cloud = point_cloud[point_cloud[:, 2] > negative_filter]  # points that have height larger than negative_filter

    return point_cloud

def visualize_results_rbw(dump_dir, predictions, rbw_mesh, object_point_threshold=200):
    # confident_bboxes = os.path.join(dump_dir, "0000_pred_confident_nms_bbox.ply")
    # bboxes = o3d.io.read_triangle_mesh(confident_bboxes)
    # bboxes.compute_vertex_normals()

    # bboxes.scale(1.2, center=bboxes.get_center())
    pcds = [rbw_mesh]
    for box in predictions[0]:

        #transformations needed to translate votenet coordinates to NORMAL
        bounding_box = np.array(box[1])
        bounding_box[:, [0, 1, 2]] = bounding_box[:, [0, 2, 1]]
        bounding_box[:,2] = bounding_box[:,2] * -1

        box3d = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box))
        box3d.scale(1.05, center=box3d.get_center())
        box3d.color = np.array([1., 0., 0.])

        in_box_vertices = np.asarray(box3d.get_point_indices_within_bounding_box(rbw_mesh.vertices))

        if len(in_box_vertices) < object_point_threshold: continue
        # colors = np.asarray(rbw_mesh.vertex_colors)
        # colors[in_box_vertices] = [1,0,0]
        # rbw_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        msh = o3d.geometry.TriangleMesh.select_by_index(rbw_mesh, in_box_vertices)
        msh.paint_uniform_color([1,0,0])
        msh.compute_vertex_normals()
        msh.compute_triangle_normals()
        # msh.scale(1.1, center=msh.get_center())

        pcds.append(box3d)
        pcds.append(msh)

    o3d.visualization.draw_geometries(pcds)

def visualize_results(dump_dir, predictions):

    files = os.listdir(dump_dir)
    confident_bboxes = [file for file in files if file.split('_',maxsplit=1)[1] == "pred_confident_nms_bbox.ply"]
    pcs = [file for file in files if file.split('_',maxsplit=1)[1] == "pc.ply"]
    # pcs_w_label = [file for file in files if file.split('_',maxsplit=1)[1] == "proposal_pc_objectness_label.ply"]

    confident_bboxes.sort()
    pcs.sort()
    # pcs_w_label.sort()

    pc = o3d.io.read_point_cloud(os.path.join(dump_dir, pcs[0]))
    bbox = o3d.io.read_triangle_mesh(os.path.join(dump_dir, confident_bboxes[0]))
    bbox.compute_vertex_normals()
    # pc_w_label = o3d.io.read_point_cloud(os.path.join(dump_dir, pcs_w_label[0]))

    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd.compute_vertex_normals()
    # print(np.asarray(pcd.triangles))
    pcds = [pc, bbox]

    #recolor white voxels if it is a pointcloud
    # if hasattr(pcd, 'points'):
    #     colors = np.asarray(pcd.colors)
    #     white_idx =np.where((colors[:, 0] == 1) & (colors[:, 1] == 1) & (colors[:, 2] == 1))[0]
    #     colors[white_idx, :] = [0, 0, 0]
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    #
    # if config.gt_path is not None:
    #     ground_truth = o3d.io.read_point_cloud(config.gt_path)
    #     ground_truth.paint_uniform_color([1, 0, 0])
    #     pcds.append(ground_truth)

    for box in predictions[0]:

        #transformations needed to translate votenet coordinates to NORMAL
        bounding_box = np.array(box[1])
        bounding_box[:, [0, 1, 2]] = bounding_box[:, [0, 2, 1]]
        bounding_box[:,2] = bounding_box[:,2] * -1

        box3d = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box))
        box3d_center = box3d.get_center()
        box3d_confidence = box[2]

        box_text = text_3d(str(round(box3d_confidence,5)), font_size=40)
        box_text.translate([box3d_center[0], box3d_center[1], box3d.get_max_bound()[2]+0.3])
        box_text.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0,np.pi/2,0]), box_text.get_center())

        pcds.append(box_text)
        pcds.append(box3d)

    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcds.append(coords)


    o3d.visualization.draw_geometries(pcds)


def text_3d(text, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """

    from PIL import Image, ImageFont, ImageDraw

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)
    return pcd


if __name__=='__main__':
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')

    elif FLAGS.dataset == 'panelnet':
        sys.path.append(os.path.join(ROOT_DIR, 'panelnet'))
        from panel_dataset import DC # dataset config
        from panel_data import produce_unseen_sample_as_np
        panel_train_dir = os.path.join(BASE_DIR, FLAGS.checkpoint_dir)
        checkpoint_path = os.path.join(panel_train_dir, 'checkpoint.tar')
        if os.path.isfile(FLAGS.pc_path):
            pc_path = FLAGS.pc_path
        else:
            print("Pointcloud not found, using generated one")
            pc_path = 'generate'
    else:
        print('Unkown dataset %s. Exiting.'%(DATASET))
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
        'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms,
                   'per_class_proposal': FLAGS.per_class_proposal,
                   'conf_thresh': FLAGS.conf_thresh, 'dataset_config': DC}


    # Init the model and optimzier
    # MODEL = importlib.import_module('votenet') # import network module
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = MODEL.VoteNet(num_proposal=256, input_feature_dim=int(FLAGS.use_height)*1, vote_factor=1,
    #     sampling=FLAGS.cluster_sampling, num_class=DC.num_class,
    #     num_heading_bin=DC.num_heading_bin,
    #     num_size_cluster=DC.num_size_cluster,
    #     mean_size_arr=DC.mean_size_arr).to(device)

    MODEL = importlib.import_module(FLAGS.model)  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = int(FLAGS.use_height)*1

    if FLAGS.model == 'boxnet':
        Detector = MODEL.BoxNet
    else:
        Detector = MODEL.VoteNet

    net = Detector(num_class=DC.num_class,
                   num_heading_bin=DC.num_heading_bin,
                   num_size_cluster=DC.num_size_cluster,
                   mean_size_arr=DC.mean_size_arr,
                   num_proposal=FLAGS.num_target,
                   input_feature_dim=num_input_channel,
                   vote_factor=FLAGS.vote_factor,
                   sampling=FLAGS.cluster_sampling).to(device)

    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))

    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    if pc_path == 'generate':
        pcd = produce_unseen_sample_as_np(FLAGS.num_point)
        pc = preprocess_point_cloud(pcd)

    else:
        if FLAGS.rbw:
            rbw_mesh = o3d.io.read_triangle_mesh(pc_path)
            print('RoboWeldAR mesh loaded: %s' % (pc_path))
            point_cloud = preprocess_rbw_mesh(rbw_mesh)
        else:
            point_cloud = read_ply(pc_path)
            print('Loaded point cloud data: %s' % (pc_path))

        pc = preprocess_point_cloud(point_cloud)

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    print(inputs)
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
    for box in pred_map_cls[0]:
        print("bbox class: ", box[0], " bbox confidence: ", box[2] )

    dump_dir = os.path.join(ROOT_DIR, '%s_inference_results'%(FLAGS.dataset))
    if not os.path.exists(dump_dir): os.mkdir(dump_dir)
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))
    if FLAGS.rbw:
        visualize_results_rbw(dump_dir, pred_map_cls, rbw_mesh)
    else:
        visualize_results(dump_dir, pred_map_cls)





