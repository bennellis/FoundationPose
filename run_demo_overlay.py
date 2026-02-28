#!/usr/bin/env python
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from estimater import *
from datareader import *
import argparse


def make_highlight_mesh(mesh, color=(255, 215, 0)):
  mesh = mesh.copy()
  colors = np.tile(np.array(color, dtype=np.uint8).reshape(1, 3), (len(mesh.vertices), 1))
  mesh.visual.vertex_colors = colors
  return mesh


def overlay_render(rgb, render_rgb, render_depth, alpha=0.6):
  out = rgb.astype(np.float32)
  mask = render_depth > 0
  if mask.any():
    out[mask] = out[mask] * (1.0 - alpha) + render_rgb[mask] * alpha
  return out.clip(0, 255).astype(np.uint8)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--overlay_alpha', type=float, default=0.6)
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  mesh_highlight = make_highlight_mesh(mesh)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/mesh_overlay {debug_dir}/ob_in_cam')

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      H, W = color.shape[:2]
      ob_in_cams = torch.as_tensor(pose.reshape(1, 4, 4), device='cuda', dtype=torch.float)
      render_rgb, render_depth, _ = nvdiffrast_render(
        K=reader.K,
        H=H,
        W=W,
        ob_in_cams=ob_in_cams,
        glctx=glctx,
        mesh=mesh_highlight,
        use_light=True,
      )
      render_rgb = (render_rgb[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
      render_depth = render_depth[0].detach().cpu().numpy()
      vis = overlay_render(color, render_rgb, render_depth, alpha=args.overlay_alpha)
      os.makedirs(f'{debug_dir}/mesh_overlay', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/mesh_overlay/{reader.id_strs[i]}.png', vis)
