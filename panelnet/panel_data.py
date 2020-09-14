import open3d as o3d
import numpy as np
import copy
import json
import os
import time

CLASS_WHITELIST = ['panel', 'floor']

class LabelBox:

    def __init__(self, center, size, heading_angle, classname, bounding_box):
        self.type2class = {'panel': 0, 'floor': 1}
        self.center = center
        self.size = size
        self.heading_angle = heading_angle
        self.classname = classname
        self.type = self.type2class[self.classname]
        self.bounding_box = bounding_box


class PanelConfig:

    def __init__(self, translation=(0,0,0), rotation= (0,0,0), size=(0.1,2,1)): #x,y,z
        self.translation = translation
        self.rotation = rotation
        self.size = size

    def randomize(self,translation_radius=1):
        self.translation = (np.random.uniform(-translation_radius, translation_radius), np.random.uniform(-translation_radius, translation_radius), 0)
        self.rotation = (0,0, np.random.uniform(0, 2*np.pi))
        self.size = (np.random.uniform(0.01,0.1), np.random.uniform(1, 3), np.random.uniform(0.5, 1.5))

def switch_xy(size: np.array):
    return np.array([size[1], size[0], size[2]])

def introduce_noise(mesh, noise=2.0):
    vertices = np.asarray(mesh.vertices)
    vertices += np.random.uniform(-noise, noise, size=vertices.shape)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

def create_panel(size=(1,1,1), sub_iter=4, noise = 0.01 ):
    box = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
    mesh = box.subdivide_midpoint(sub_iter) #resample
    introduce_noise(mesh, noise)
    mesh.compute_vertex_normals()
    return mesh


def generate_welding_area(panels_num=2, vis=False):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()

    panels = []
    labelboxes = []
    for i in range(panels_num):
        #creation
        panel_config = PanelConfig()
        panel_config.randomize()
        panel = create_panel(panel_config.size, sub_iter=4, noise = 0.01)
        panel.paint_uniform_color(list(np.random.random(3)))
        #transformations
        panel.rotate(o3d.geometry.get_rotation_matrix_from_xyz(panel_config.rotation), panel.get_center())
        panel.translate(panel_config.translation)

        #bounding label box
        oriented_bounding_box = panel.get_oriented_bounding_box()
        center_oriented = oriented_bounding_box.get_center()
        labelbox = LabelBox(center_oriented, np.array(panel_config.size)/2, -panel_config.rotation[2], CLASS_WHITELIST[0], oriented_bounding_box)

        panels.append(panel)
        labelboxes.append(labelbox)

    mesh = panels[0]
    for panel in panels[1:]: mesh += panel

    if vis: o3d.visualization.draw_geometries([coords, mesh], mesh_show_wireframe=True)

    # return mesh, [labelbox, labelbox2, labelbox3]
    return mesh, labelboxes


def export(idx: int, mesh, labelboxes: [LabelBox], output_folder, n_points, export_mesh=True):
    #--------mesh----------
    if export_mesh:
        o3d.io.write_triangle_mesh(os.path.join(output_folder,"%04d_mesh.ply" % idx), mesh, compressed=True, write_vertex_colors=True)

    #-----pointcloud---------
    # sampled_pointcloud = mesh.sample_points_poisson_disk(number_of_points=int(n_points), init_factor=10,  use_triangle_normal=True) # better sampling
    sampled_pointcloud = mesh.sample_points_uniformly(number_of_points=int(n_points),  use_triangle_normal=True) # quicker sampling

    # add ground floor as noise

    floor_noise = produce_floor_noise()
    pc = np.vstack((np.asarray(sampled_pointcloud.points), floor_noise))

    np.savez_compressed(os.path.join(output_folder, '%04d_pc.npz' % idx), pc=pc)

    #--------bbox/labelbox--------
    object_list = []
    for lb in labelboxes:
        if lb.classname not in CLASS_WHITELIST: continue
        obb = np.zeros((8))
        obb[0:3] = lb.center
        obb[3:6] = lb.size
        obb[6] = lb.heading_angle
        obb[7] = lb.type
        object_list.append(obb)
    if len(object_list) == 0:
        obbs = np.zeros((0, 8))
    else:
        obbs = np.vstack(object_list)  # (K,8)

    np.save(os.path.join(output_folder, '%04d_bbox.npy' % (idx)), obbs)

    #---------save votes----------
    N = len(pc)
    point_votes = np.zeros((N, 10))  # 3 votes and 1 vote mask
    point_vote_idx = np.zeros((N)).astype(np.int32)  # in the range of [0,2]
    indices = np.arange(N)

    for lb in labelboxes:
        if lb.classname not in CLASS_WHITELIST: continue
        try:
            # Find all points in this object's OBB
            box3d_pts_inds = lb.bounding_box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pc))
            # Assign first dimension to indicate it is in an object box
            point_votes[box3d_pts_inds, 0] = 1
            pc_in_box3d = pc[box3d_pts_inds]
            # Add the votes (all 0 if the point is not in any object's OBB)
            votes = np.expand_dims(lb.center, 0) - pc_in_box3d[:, 0:3]
            sparse_inds = indices[box3d_pts_inds]  # turn dense True,False inds to sparse number-wise inds
            for i in range(len(sparse_inds)):
                j = sparse_inds[i]
                point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
                # Populate votes with the fisrt vote
                if point_vote_idx[j] == 0:
                    point_votes[j, 4:7] = votes[i, :]
                    point_votes[j, 7:10] = votes[i, :]
            point_vote_idx[box3d_pts_inds] = np.minimum(2, point_vote_idx[box3d_pts_inds] + 1)
        except:
            print('ERROR ----', idx, lb.classname)

    np.savez_compressed(os.path.join(output_folder, '%04d_votes.npz' % (idx)), point_votes=point_votes)

def produce_floor_noise(floor_radius=3, z_var=0.01, points_num=15000):
    # add ground floor as noise
    return np.random.uniform([-floor_radius, -floor_radius, -z_var], [floor_radius, floor_radius, z_var], [points_num, 3])

def produce_unseen_sample_as_np(n_points):
    print(f'Producing unseen sample...')
    mesh, labelboxes = generate_welding_area()
    pt = mesh.sample_points_uniformly(number_of_points=int(n_points*3/4),  use_triangle_normal=True)

    floor_noise = produce_floor_noise(points_num=int(n_points/4))
    pcd = np.vstack((np.asarray(pt.points), floor_noise))
    print(len(pcd))
    # o3d.io.write_point_cloud(os.path.join(output_folder, "unseen_pointcloud.ply"), pcd)
    return pcd

if __name__ == '__main__':
    items_gen_num = 12000
    train2test_ratio = 0.6
    output_folder= "dataset"
    train_output_folder = "dataset/panel_data_train"
    val_output_folder = "dataset/panel_data_val"
    n_points = 40000
    overwrite = False
    train_ids = []
    val_ids = []
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)
    else:
        train_dirs = os.listdir(train_output_folder)
        train_ids = [int(name[0:4]) for name in train_dirs]
    if not os.path.exists(val_output_folder):
        os.mkdir(val_output_folder)
    else:
        val_dirs = os.listdir(val_output_folder)
        val_ids = [int(name[0:4]) for name in val_dirs]

    #DEBUG
    # mesh, labelboxes = generate_welding_area(vis=True)
    # export(0, mesh, labelboxes, train_output_folder, 50000)
    # exit()

    #training
    print("---------Training dataset----------")
    for i in range(int(items_gen_num * train2test_ratio)):
        if (not overwrite) and (i in train_ids): continue
        start = time.time()
        print("Generating welding area no.", i,"...")
        mesh, labelboxes = generate_welding_area()
        print(f'Exporting welding area...')
        export(i, mesh, labelboxes, train_output_folder, n_points, export_mesh=False)
        end = time.time()
        print("Welding area no.", i, "completed in ",end - start, "seconds \n")

    #testing
    print(" \n \n---------Validation dataset---------- \n")
    for i in range(int(items_gen_num * (1 - train2test_ratio))+1):
        if (not overwrite) and (i in val_ids): continue
        start = time.time()
        print("Generating welding area no.", i,"... \n")
        mesh, labelboxes = generate_welding_area()
        print(f'Exporting welding area...')
        export(i, mesh, labelboxes, val_output_folder, n_points, export_mesh=False)
        end = time.time()
        print("Welding area no.", i, "completed in ", end - start, "seconds \n")
