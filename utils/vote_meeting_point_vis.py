import open3d as o3d
import os, sys, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir',  help='Directory that includes the seeds points and prediction points')
parser.add_argument('--file_no',  help='Number of file to be visualized')

FLAGS = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))





if __name__=='__main__':
    # Set file paths and dataset config
    dump_dir = os.path.abspath(os.path.join(BASE_DIR, "..",FLAGS.dump_dir))
    seed_path = os.path.join(dump_dir, FLAGS.file_no + "_seed_pc.ply")
    vgen_path = os.path.join(dump_dir, FLAGS.file_no + "_vgen_pc.ply")
    pc_path = os.path.join(dump_dir, FLAGS.file_no + "_pc.ply")
    conf_path = os.path.join(dump_dir, FLAGS.file_no + "_confident_proposal_pc.ply")

    # pcd = o3d.geometry.PointCloud()
    # pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    # pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    seed_pcd = o3d.io.read_point_cloud(seed_path)
    vgen_pcd = o3d.io.read_point_cloud(vgen_path)
    pcd = o3d.io.read_point_cloud(pc_path)
    confident_proposal_pcd = o3d.io.read_point_cloud(conf_path)


    seed_pcd.paint_uniform_color([0,1,0])
    vgen_pcd.paint_uniform_color([1,0,0])
    pcd.paint_uniform_color([0.9,0.9,0.9])
    confident_proposal_pcd.paint_uniform_color([0,0,1])

    print(np.asarray(seed_pcd.points).shape)
    print(np.asarray(vgen_pcd.points).shape)
    print(np.asarray(confident_proposal_pcd.points))

    ar =  np.arange(np.asarray(seed_pcd.points).shape[0])
    correspondences = [(i,i) for i in ar]

    lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(seed_pcd,vgen_pcd, correspondences)

    o3d.visualization.draw_geometries([seed_pcd,vgen_pcd, pcd,lines])
