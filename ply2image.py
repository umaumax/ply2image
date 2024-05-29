#!/usr/bin/env python3

import argparse
import os
import copy

import open3d as o3d
import numpy as np
from PIL import Image


def rotate_pcd_upside_down(pcd):
    rotate_pcd(pcd, 180.0, [1, 0, 0])


def rotate_pcd(pcd, rotation_degree_angle, rotation_axis):
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.deg2rad(rotation_degree_angle) * np.array(rotation_axis))
    pcd.rotate(rotation_matrix, center=pcd.get_center())


def create_aabb_and_coordinate_frame(pcd, aabb_color=(1, 0, 0)):
    """
    Args:
      pcd: Open3D point cloud
      aabb_color: axis_aligned_bounding_box color (default: red)
    """
    aabb = pcd.get_axis_aligned_bounding_box()

    extent = aabb.get_extent()
    length, width, height = extent

    average_size = np.mean([length, width, height])

    center = aabb.get_center()

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=average_size / 2,
        origin=center
    )

    return (aabb, coordinate_frame)


def generate_images(pcd, num_images, output_path,
                    width=1200, height=800,
                    background_color=[0, 0, 0],  # black
                    zoom=1.0,
                    degree=10.0,
                    image_resize_factor=1.0, image_list=[0], coordinate_frame_flag=False):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.get_render_option().background_color = background_color

    orig_pcd = copy.copy(pcd)
    images = []
    params = None
    for i in range(num_images):
        pcd = copy.copy(orig_pcd)

        rotation_degree_angle = degree * i
        rotation_axis = [0, 1, 0]
        rotate_pcd(pcd, rotation_degree_angle, rotation_axis)

        geometries = []
        if coordinate_frame_flag:
            aabb, coordinate_frame = create_aabb_and_coordinate_frame(pcd)
            geometries += [aabb, coordinate_frame]

        vis.add_geometry(pcd)
        for geometry in geometries:
            vis.add_geometry(geometry)

        ctr = vis.get_view_control()
        if i == 0:
            params = ctr.convert_to_pinhole_camera_parameters()
            print('camera extrinsic params:\n', params.extrinsic)
        # reset the first camera setting
        ctr.convert_from_pinhole_camera_parameters(
            params, allow_arbitrary=True)
        ctr.set_zoom(zoom)

        vis.poll_events()
        vis.update_renderer()

        if i in image_list:
            image_path = os.path.join(output_path, f'image_{i:03}.png')
            print(image_path)
            vis.capture_screen_image(image_path)

        img = vis.capture_screen_float_buffer()
        image = np.asarray(img) * 255  # 0-1 to 0-255
        image = Image.fromarray(image.astype(np.uint8))
        if image_resize_factor != 1.0:
            image = image.resize(
                (int(width * image_resize_factor), int(height * image_resize_factor)))
        images.append(image)

        vis.clear_geometries()

    vis.destroy_window()
    return images


def main():
    parser = argparse.ArgumentParser(
        description='Generate rotating images of a 3D object')
    parser.add_argument('input_ply', type=str,
                        help='Path to the input PLY file')
    parser.add_argument('--output-dir', type=str, default='output_images',
                        help='Directory to save the generated images')
    parser.add_argument('--width', type=int, default=1200,
                        help='Width of images')
    parser.add_argument('--height', type=int, default=800,
                        help='Height of images')
    parser.add_argument('--num-images', type=int, default=36,
                        help='Number of images to generate')
    parser.add_argument('--degree', type=float, default=10.0,
                        help='Set degree of GIT animation')
    parser.add_argument('--resize-factor', type=float, default=0.5,
                        help='Resize factor for output images (0-1)')
    parser.add_argument('--zoom', type=float, default=1.0,
                        help='Zoom parameter 0.0(close) <-> far')
    parser.add_argument('--gif-duration', type=float, default=1000.0 / 36.0,
                        help='Duration of gif animation')
    parser.add_argument(
        '--output-gif',
        default='output_animation.gif',
        help='Output GIF filepath')
    parser.add_argument(
        '--coordinate',
        action='store_true',
        help='Set coordinate object')
    parser.add_argument(
        '--upside-down',
        action='store_true',
        help='Rotate model to upside-down')
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Launch 3D interactive viewer')
    parser.add_argument(
        '--image-list',
        nargs='*',
        type=int,
        default=None,
        help='Save the image of the listed index')
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input_ply)
    if args.upside_down:
        rotate_pcd_upside_down(pcd)

    if args.interactive:
        geometries = [pcd]
        if args.coordinate:
            aabb, coordinate_frame = create_aabb_and_coordinate_frame(pcd)
            geometries += [aabb, coordinate_frame]

        o3d.visualization.draw_geometries(geometries)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    image_list = args.image_list
    if image_list is None:
        image_list = list(range(args.num_images))
    images = generate_images(
        pcd, args.num_images, args.output_dir,
        width=args.width, height=args.height,
        zoom=args.zoom,
        degree=args.degree,
        image_resize_factor=args.resize_factor, image_list=image_list, coordinate_frame_flag=args.coordinate)

    if args.output_gif:
        print('Generating GIF...')
        image_path = args.output_gif
        images[0].save(image_path,
                       save_all=True,
                       append_images=images[1:],
                       optimize=True,
                       duration=args.gif_duration,
                       loop=0)
        print(f'GIF saved to {image_path}')


if __name__ == '__main__':
    main()
