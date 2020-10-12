import open3d as o3d
import numpy as np
import copy
import json
import os
import time

CLASS_WHITELIST = ['panel', 'hpanel']

class LabelBox:
    def __init__(self, center, size, heading_angle, classname, bounding_box):
        self.type2class = {'panel': 0, 'hpanel': 1}
        self.center = center
        self.size = size
        self.heading_angle = heading_angle
        self.classname = classname
        self.type = self.type2class[self.classname]
        self.bounding_box = bounding_box


class PanelConfig:
    def __init__(self, position=(0,0,0), rotation= (0,0,0), size=(0.1,2,1)): #x,y,z
        self.position = position
        self.rotation = rotation
        self.size = size

    def randomize(self,position_radius=0.2, is_hpanel=False):
        self.position = (np.random.uniform(-position_radius, position_radius), np.random.uniform(-position_radius, position_radius), 0)
        self.rotation = (0,0, np.random.uniform(0, np.pi))#2*np.pi))
        if not is_hpanel: #panel
            self.size = (np.random.uniform(0.01, 0.013), np.random.uniform(0.275, 0.325), np.random.uniform(0.175, 0.225))
        else:
            self.size = (np.random.uniform(0.175, 0.225), np.random.uniform(0.275, 0.325), np.random.uniform(0.01, 0.013))


def switch_xy(size: np.array):
    return np.array([size[1], size[0], size[2]])

def introduce_noise(mesh, noise=2.0):
    vertices = np.asarray(mesh.vertices)
    vertices += np.random.uniform(-noise, noise, size=vertices.shape)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

def create_panel(size=(1,1,1), sub_iter=4, noise = 0.01 ):
    box = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
    # box.triangles = o3d.utility.Vector3iVector(np.asarray(box.triangles)[:10, :])
    # box.triangle_normals = o3d.utility.Vector3dVector(np.asarray(box.triangle_normals)[:10, :])
    # o3d.visualization.draw_geometries([box])
    mesh = box.subdivide_midpoint(sub_iter) #resample
    introduce_noise(mesh, noise)
    mesh.compute_vertex_normals()
    return mesh, sub_iter


def create_surface(size=(0.75, 0.75, 0), sub_iter=4, noise = 0.02):
    sur = o3d.geometry.TriangleMesh()
    sur.vertices = o3d.utility.Vector3dVector([[0,0,0], [size[0],0,0], [0,size[1],0], size])
    sur.triangles = o3d.utility.Vector3iVector([[0,1,3], [3,2,0]])

    mesh = sur.subdivide_midpoint(sub_iter) #resample
    introduce_noise(mesh, noise)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh

def get_idx_list(pointcloud,panels,labelboxes):
    color_list = []
    for i in np.asarray(pointcloud.colors)[:,0]:
        if np.round(i,4) not in color_list:
            color_list.append(np.round(i,4)) 
    
    #take list of ending point of every object
    end_of_object_list=[]
    for i in range(len(panels)+1):
        idx = np.where(np.asarray(pointcloud.colors)[:,0] == color_list[i])
        end_of_object_list.append(idx[0][-1])
    
    #create a pointcloud only with surface point
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(pointcloud.points)[:end_of_object_list[0], :])

    #create a list with indexes of points that have to be removed 
    indexes = []
    for i in range(len(panels)):
        obb = copy.deepcopy(labelboxes[i].bounding_box)
        obb.translate(np.array([0,0,-0.005]))
        indexes.append(obb.get_point_indices_within_bounding_box(surface_pcd.points))

    idx_list =[]
    for i in range(len(panels)):
        idx_list += indexes[i] 

    new = sorted(idx_list)

    newlist = []

    for i in new:
        if i not in newlist:
            newlist.append(i)

    newlist.reverse()
    
    #seting the reconstructed pointcloud
    for i in newlist:
        pointcloud.points = o3d.utility.Vector3dVector(np.delete(np.asarray(pointcloud.points), i,0))

    return pointcloud


def generate_welding_area(panels_num=2, hpanels_num=1, vis=False):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()

    panels = []
    labelboxes = []

    #hpannels
    cnt=0
    while cnt != hpanels_num:
        panel_config = PanelConfig()
        panel_config.randomize(is_hpanel=True)
        panel, sub_iter = create_panel(panel_config.size, sub_iter=4, noise = 0.002)
        panel.paint_uniform_color(list(np.random.randint(10000, size=3)/10000))
        oriented_bounding_box = panel.get_axis_aligned_bounding_box().get_oriented_bounding_box()
        
        #transformations
        panel.rotate(o3d.geometry.get_rotation_matrix_from_xyz(panel_config.rotation), panel.get_center())
        panel.translate(np.array(panel_config.position) - np.array([panel.get_center()[0], panel.get_center()[1], 0]))
        #
        oriented_bounding_box.rotate(o3d.geometry.get_rotation_matrix_from_xyz(panel_config.rotation), panel.get_center())
        oriented_bounding_box.translate(np.array(panel_config.position) - np.array([panel.get_center()[0], panel.get_center()[1], 0]))
        #
        #bounding label box
        center_oriented = oriented_bounding_box.get_center()
        labelbox = LabelBox(center_oriented, (np.array(panel_config.size)+0.005)/2, -panel_config.rotation[2], CLASS_WHITELIST[1], oriented_bounding_box)
        #
        #deleting bottom layer
        panel.triangles = o3d.utility.Vector3iVector(np.asarray(panel.triangles)[:(10*(4**sub_iter)), :])
        panel.triangle_normals = o3d.utility.Vector3dVector(np.asarray(panel.triangle_normals)[:(10*(4**sub_iter)), :])
        #
        if cnt == 0:
            panels.append(panel)
            labelboxes.append(labelbox)
            cnt += 1
        if cnt == 1 and not panels[cnt-1].is_intersecting(panel):
            cnt += 1
            panels.append(panel)
            labelboxes.append(labelbox)
        if cnt == 2 and not panels[cnt-1].is_intersecting(panel) and not panels[cnt-2].is_intersecting(panel):
            cnt += 1
            panels.append(panel)
            labelboxes.append(labelbox)
        else:print("fail to find")
        
    #verticsal panels
    #bounding label box
    for i in range(panels_num):
        #creation
        panel_config = PanelConfig()
        panel_config.randomize()
        panel, sub_iter = create_panel(panel_config.size, sub_iter=4, noise = 0.002)
        panel.paint_uniform_color(list(np.random.randint(10000, size=3)/10000))
        oriented_bounding_box = panel.get_axis_aligned_bounding_box().get_oriented_bounding_box()

        #transformations
        panel.rotate(o3d.geometry.get_rotation_matrix_from_xyz(panel_config.rotation), panel.get_center())
        #panels are not centered, and because we want the position relative to the axis center, we subtract the object x-y center from the number
        panel.translate(np.array(panel_config.position) - np.array([panel.get_center()[0], panel.get_center()[1], 0]))
        oriented_bounding_box.rotate(o3d.geometry.get_rotation_matrix_from_xyz(panel_config.rotation), panel.get_center())
        oriented_bounding_box.translate(np.array(panel_config.position) - np.array([panel.get_center()[0], panel.get_center()[1], 0]))

        center_oriented = oriented_bounding_box.get_center()
        labelbox = LabelBox(center_oriented, (np.array(panel_config.size)+0.005)/2, -panel_config.rotation[2], CLASS_WHITELIST[0], oriented_bounding_box)
        
        #deleting bottom layer
        panel.triangles = o3d.utility.Vector3iVector(np.asarray(panel.triangles)[:(10*(4**sub_iter)), :])
        panel.triangle_normals = o3d.utility.Vector3dVector(np.asarray(panel.triangle_normals)[:(10*(4**sub_iter)), :])
        
        panels.append(panel)
        labelboxes.append(labelbox)


    floor = create_surface(size=(0.75,0.75,0), noise=0.005)
    floor.paint_uniform_color([0.5,0.5,0.5])
    floor.translate([-floor.get_center()[0], -floor.get_center()[1], 0]) #axis center

    mesh = copy.deepcopy(floor)

    for panel in panels: mesh += panel

    pcd = mesh.sample_points_uniformly(number_of_points=30000)

    #reconstruct the final pointcloud
    pcd = get_idx_list(pcd,panels,labelboxes)
      
    o3d.visualization.draw_geometries([pcd])
    
    if vis: o3d.visualization.draw_geometries([coords, mesh])

    # return mesh, [labelbox, labelbox2, labelbox3]
    return pcd, mesh, labelboxes


def export(pcd, idx: int, mesh, labelboxes: [LabelBox], output_folder, n_points, export_mesh=True):
    #--------mesh----------
    if export_mesh:
        o3d.io.write_triangle_mesh(os.path.join(output_folder,"%04d_mesh.ply" % idx), mesh, compressed=True, write_vertex_colors=True)

    #-----pointcloud---------
    # sampled_pointcloud = mesh.sample_points_poisson_disk(number_of_points=int(n_points), init_factor=10,  use_triangle_normal=True) # better sampling
    # sampled_pointcloud = mesh.sample_poinsts_uniformly(number_of_points=int(n_points),  use_triangle_normal=True) # quicker sampling

    pc = np.asarray(pcd.points)

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

def produce_floor_noise(floor_radius=0.4, z_var=0.01, points_num=15000):
    # add ground floor as noise
    return np.random.uniform([-floor_radius, -floor_radius, -z_var], [floor_radius, floor_radius, z_var], [points_num, 3])

def produce_random_noise(radius=2, points_num=2000):
    # add general noise
    return np.random.uniform([-radius, -radius, 0], [radius, radius, radius], [points_num, 3])

def produce_unseen_sample_as_np(n_points):
    print(f'Producing unseen sample...')
    # mesh, labelboxes = generate_welding_area()
    mesh, labelboxes = generate_welding_area(panels_num=np.random.randint(2, 5), hpanels_num=np.random.randint(1, 3))
    pt = mesh.sample_points_uniformly(number_of_points=int(n_points),  use_triangle_normal=True)

    return np.asarray(pt.points)

if __name__ == '__main__':
    items_gen_num = 10
    train2val_ratio = 1
    output_folder= "dataset"
    train_output_folder = "dataset/panel_data_train"
    val_output_folder = "dataset/panel_data_val"
    test_output_folder = "dataset/panel_data_test"
    n_points = 30000
    overwrite = False
    train_ids = []
    val_ids = []
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    #training
    if not os.path.exists(train_output_folder):
        os.mkdir(train_output_folder)
    else:
        train_dirs = os.listdir(train_output_folder)
        train_ids = [int(name[0:4]) for name in train_dirs]
    #val
    if not os.path.exists(val_output_folder):
        os.mkdir(val_output_folder)
    else:
        val_dirs = os.listdir(val_output_folder)
        val_ids = [int(name[0:4]) for name in val_dirs]
    #testing
    if not os.path.exists(test_output_folder):
        os.mkdir(test_output_folder)
    else:
        test_dirs = os.listdir(test_output_folder)
        test_ids = [int(name[0:4]) for name in test_dirs]

    # DEBUG
    pcd, mesh, labelboxes = generate_welding_area(panels_num=np.random.randint(1,4), hpanels_num=np.random.randint(1,3), vis=False)
    # pcd, mesh, labelboxes = generate_welding_area(panels_num=0, hpanels_num=3, vis=False)
    # export(0, mesh, labelboxes, train_output_folder, 50000)
    exit()

    #training
    print("---------Training dataset----------")
    for i in range(int(items_gen_num * train2val_ratio)):
        if (not overwrite) and (i in train_ids): continue
        start = time.time()
        print("Generating welding area no.", i,"...")
        pcd, mesh, labelboxes = generate_welding_area(panels_num=np.random.randint(2,4), hpanels_num=np.random.randint(1,3))
        print(f'Exporting welding area...')
        export(pcd, i, mesh, labelboxes, train_output_folder, n_points, export_mesh=False)
        end = time.time()
        print("Welding area no.", i, "completed in ",end - start, "seconds \n")

    #validation
    print(" \n \n---------Validation dataset---------- \n")
    for i in range(int(items_gen_num * (1 - train2val_ratio)) + 1):
        if (not overwrite) and (i in val_ids): continue
        start = time.time()
        print("Generating welding area no.", i,"... \n")
        pcd, mesh, labelboxes = generate_welding_area(panels_num=np.random.randint(2,4), hpanels_num=np.random.randint(1,3))
        print(f'Exporting welding area...')
        export(pcd, i, mesh, labelboxes, val_output_folder, n_points, export_mesh=False)
        end = time.time()
        print("Welding area no.", i, "completed in ", end - start, "seconds \n")

    #testing
    # print(" \n \n---------Testing dataset---------- \n")
    # for i in range(int(items_gen_num * (1 - train2val_ratio)) + 1):
    #     if (not overwrite) and (i in test_ids): continue
    #     start = time.time()
    #     print("Generating welding area no.", i,"... \n")
    #     mesh, labelboxes = generate_welding_area(panels_num=np.random.randint(2,5), hpanels_num=np.random.randint(1,3))
    #     print(f'Exporting welding area...')
    #     export(i, mesh, labelboxes, test_output_folder, n_points, export_mesh=True)
    #     end = time.time()
    #     print("Welding area no.", i, "completed in ", end - start, "seconds \n")