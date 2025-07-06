import numpy as np
import open3d as o3d
import cv2
import time
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import argparse
from datetime import datetime
import subprocess
import os
from PIL import Image

class Open3DTexturedVideoCreator:
    """Open3Dを使用したテクスチャ付き3Dモデルの動画作成クラス（テクスチャ修正版）"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Output/open3d_textured_video_output_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # フレーム設定
        self.temp_frame_dir = None
        
        # Open3Dビジュアライザー設定
        self.vis = None
        self.mesh = None
        self.camera_params = None
        
    def parse_mtl_file(self, mtl_path: Path) -> Dict:
        """MTLファイルを解析してマテリアル情報を取得"""
        materials = {}
        current_material = None
        
        try:
            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('newmtl'):
                        current_material = line.split()[1]
                        materials[current_material] = {}
                    elif current_material and line.startswith('map_Kd'):
                        # Diffuse texture map
                        texture_file = line.split()[1]
                        materials[current_material]['diffuse_texture'] = texture_file
                    elif current_material and line.startswith('map_Ka'):
                        # Ambient texture map
                        texture_file = line.split()[1]
                        materials[current_material]['ambient_texture'] = texture_file
                    elif current_material and line.startswith('Kd'):
                        # Diffuse color
                        r, g, b = map(float, line.split()[1:4])
                        materials[current_material]['diffuse_color'] = [r, g, b]
        except Exception as e:
            print(f"Error parsing MTL file: {e}")
        
        return materials
    
    def load_texture_image(self, texture_path: Path) -> Optional[np.ndarray]:
        """テクスチャ画像を読み込む"""
        try:
            if texture_path.exists():
                # PILで画像を読み込み、RGBに変換
                img = Image.open(texture_path).convert('RGB')
                # NumPy配列に変換
                texture_array = np.array(img)
                print(f"Loaded texture: {texture_path.name} ({texture_array.shape})")
                return texture_array
            else:
                print(f"Texture file not found: {texture_path}")
                return None
        except Exception as e:
            print(f"Error loading texture {texture_path}: {e}")
            return None
    
    def load_textured_model(self, model_path: str) -> Optional[o3d.geometry.TriangleMesh]:
        """テクスチャ付き3Dモデルを読み込む（改良版）"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist")
            return None
        
        try:
            # まずtrimeshで読み込み（テクスチャ情報を含む）
            import trimesh
            trimesh_obj = trimesh.load(str(model_path))
            
            if not hasattr(trimesh_obj, 'vertices'):
                print(f"Error: No vertices found in {model_path}")
                return None
            
            # Open3Dメッシュに変換
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
            
            # 法線を計算
            mesh.compute_vertex_normals()
            
            # MTLファイルを探す
            mtl_path = model_path.with_suffix('.mtl')
            if mtl_path.exists():
                print(f"Found MTL file: {mtl_path}")
                materials = self.parse_mtl_file(mtl_path)
                
                # テクスチャを適用
                if materials:
                    material_name = list(materials.keys())[0]  # 最初のマテリアルを使用
                    material = materials[material_name]
                    print(f"Using material: {material_name}")
                    
                    if 'diffuse_texture' in material:
                        texture_file = material['diffuse_texture']
                        # 相対パスを正しく解決
                        if os.path.isabs(texture_file):
                            texture_path = Path(texture_file)
                        else:
                            texture_path = model_path.parent / texture_file
                        
                        print(f"Looking for texture: {texture_path}")
                        texture_image = self.load_texture_image(texture_path)
                        
                        if texture_image is not None:
                            # Open3Dにテクスチャを設定
                            mesh.textures = [o3d.geometry.Image(texture_image)]
                            
                            # UV座標が必要（trimeshから取得を試行）
                            if hasattr(trimesh_obj.visual, 'uv') and trimesh_obj.visual.uv is not None:
                                # UV座標を三角形ごとに設定
                                uv_coords = trimesh_obj.visual.uv
                                # Open3DのUV座標形式に変換
                                triangle_uvs = []
                                for face in trimesh_obj.faces:
                                    face_uvs = []
                                    for vertex_idx in face:
                                        if vertex_idx < len(uv_coords):
                                            face_uvs.extend(uv_coords[vertex_idx])
                                        else:
                                            face_uvs.extend([0.0, 0.0])  # デフォルトUV
                                    triangle_uvs.extend(face_uvs)
                                
                                if len(triangle_uvs) == len(trimesh_obj.faces) * 6:  # 3頂点 × 2UV成分
                                    mesh.triangle_uvs = o3d.utility.Vector2dVector(
                                        np.array(triangle_uvs).reshape(-1, 2)
                                    )
                                    print("Applied UV coordinates for texture mapping")
                                else:
                                    print("UV coordinate count mismatch, generating default UVs")
                                    self._generate_default_uvs(mesh)
                            else:
                                print("No UV coordinates found, generating default UVs (sphere projection)")
                                print("Warning: Texture may appear distorted due to lack of original UV coordinates")
                                self._generate_default_uvs(mesh)
                            
                            print(f"Applied texture: {texture_file}")
                        else:
                            # テクスチャが読み込めない場合、頂点色を設定
                            self._apply_vertex_colors(mesh, trimesh_obj)
                    else:
                        # テクスチャがない場合、頂点色を設定
                        self._apply_vertex_colors(mesh, trimesh_obj)
                else:
                    print("No materials found in MTL file")
                    self._apply_vertex_colors(mesh, trimesh_obj)
            else:
                print("No MTL file found, applying vertex colors")
                self._apply_vertex_colors(mesh, trimesh_obj)
            
            # メッシュの基本情報を表示
            print(f"Model loaded successfully:")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Triangles: {len(mesh.triangles)}")
            print(f"  Has vertex colors: {mesh.has_vertex_colors()}")
            print(f"  Has vertex normals: {mesh.has_vertex_normals()}")
            print(f"  Has textures: {mesh.has_textures()}")
            print(f"  Has triangle UVs: {mesh.has_triangle_uvs()}")
            
            return mesh
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _generate_default_uvs(self, mesh: o3d.geometry.TriangleMesh):
        """デフォルトUV座標を生成"""
        # 簡単な球面投影UV座標を生成
        vertices = np.asarray(mesh.vertices)
        
        # 球面座標に変換
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # 球面投影UV計算
        r = np.sqrt(x**2 + y**2 + z**2)
        u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
        v = 0.5 - np.arcsin(y / r) / np.pi
        
        # NaNを修正
        u = np.nan_to_num(u, nan=0.5)
        v = np.nan_to_num(v, nan=0.5)
        
        # UV座標を[0,1]範囲に正規化
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        
        # 三角形ごとのUV座標を作成
        triangles = np.asarray(mesh.triangles)
        triangle_uvs = []
        
        for triangle in triangles:
            for vertex_idx in triangle:
                triangle_uvs.append([u[vertex_idx], v[vertex_idx]])
        
        mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
        print("Generated default spherical UV coordinates")
    
    def _apply_vertex_colors(self, mesh: o3d.geometry.TriangleMesh, trimesh_obj):
        """trimeshから頂点色をOpen3Dメッシュに適用"""
        try:
            if hasattr(trimesh_obj.visual, 'vertex_colors') and trimesh_obj.visual.vertex_colors is not None:
                # 頂点色を正規化（0-1範囲）
                vertex_colors = trimesh_obj.visual.vertex_colors[:, :3] / 255.0
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                print("Applied vertex colors from trimesh")
            else:
                # デフォルト色（薄いグレー）を設定
                default_color = np.array([[0.7, 0.7, 0.7]] * len(mesh.vertices))
                mesh.vertex_colors = o3d.utility.Vector3dVector(default_color)
                print("Applied default vertex colors")
        except Exception as e:
            print(f"Error applying vertex colors: {e}")
            # フォールバック：デフォルト色
            default_color = np.array([[0.7, 0.7, 0.7]] * len(mesh.vertices))
            mesh.vertex_colors = o3d.utility.Vector3dVector(default_color)
    
    def setup_visualizer(self, window_size: Tuple[int, int] = (1280, 720), visible: bool = False) -> bool:
        """Open3Dビジュアライザーをセットアップ（テクスチャ対応）"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="3D Textured Model Video",
                width=window_size[0],
                height=window_size[1],
                left=100,
                top=100,
                visible=visible
            )
            
            # レンダリングオプションを設定（テクスチャ表示用）
            render_option = self.vis.get_render_option()
            render_option.show_coordinate_frame = False
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # 黒い背景でテクスチャを強調
            render_option.light_on = True
            render_option.point_show_normal = False
            render_option.mesh_show_wireframe = False
            render_option.mesh_show_back_face = True
            
            return True
            
        except Exception as e:
            print(f"Error setting up visualizer: {e}")
            return False
    
    def add_model_to_visualizer(self, mesh: o3d.geometry.TriangleMesh) -> bool:
        """3Dモデルをビジュアライザーに追加（テクスチャ対応）"""
        try:
            self.mesh = o3d.geometry.TriangleMesh(mesh)
            self.vis.add_geometry(self.mesh)
            
            # カメラパラメータを取得・設定
            view_control = self.vis.get_view_control()
            
            # メッシュの境界を取得してカメラ位置を調整
            bbox = mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            max_extent = np.max(extent)
            
            # カメラを適切な距離に配置
            view_control.set_lookat(center)
            view_control.set_up([0, 1, 0])
            view_control.set_front([0, 0, 1])
            view_control.set_zoom(0.7)
            
            # 初期レンダリング
            self.vis.poll_events()
            self.vis.update_renderer()
            
            return True
            
        except Exception as e:
            print(f"Error adding model to visualizer: {e}")
            return False
    
    def create_rotation_video_with_texture(self, 
                                         model_path: str,
                                         num_frames: int = 120,
                                         rotation_axes: List[str] = ['y'],
                                         output_filename: str = None,
                                         window_size: Tuple[int, int] = (1280, 720),
                                         fps: int = 30,
                                         visible: bool = False) -> bool:
        """テクスチャ付き回転動画作成"""
        
        # モデルを読み込み
        mesh = self.load_textured_model(model_path)
        if mesh is None:
            return False
        
        # ビジュアライザーをセットアップ
        if not self.setup_visualizer(window_size, visible):
            return False
        
        # モデルをビジュアライザーに追加
        if not self.add_model_to_visualizer(mesh):
            return False
        
        # 出力ファイル名を設定
        if output_filename is None:
            model_name = Path(model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            axes_str = "_".join(rotation_axes)
            output_filename = f"{model_name}_textured_{axes_str}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # 一時フレームディレクトリを作成
        self.temp_frame_dir = self.output_dir / f"temp_frames_{timestamp}"
        self.temp_frame_dir.mkdir(exist_ok=True)
        
        print(f"Creating textured rotation video: {num_frames} frames, {rotation_axes} axes")
        print(f"Direct rendering with texture support")
        
        try:
            frame_paths = []
            frames_per_axis = num_frames // len(rotation_axes)
            
            for axis_idx, axis in enumerate(rotation_axes):
                print(f"Rendering {axis.upper()}-axis rotation...")
                
                for frame_idx in range(frames_per_axis):
                    # 回転角度を計算
                    angle_deg = (frame_idx / frames_per_axis) * 360
                    
                    # 軸に応じて回転行列を作成
                    if axis.lower() == 'x':
                        rotation_matrix = self.mesh.get_rotation_matrix_from_xyz((np.deg2rad(angle_deg), 0, 0))
                    elif axis.lower() == 'z':
                        rotation_matrix = self.mesh.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(angle_deg)))
                    else:  # Y軸
                        rotation_matrix = self.mesh.get_rotation_matrix_from_xyz((0, np.deg2rad(angle_deg), 0))
                    
                    # メッシュを初期位置にリセットして回転
                    self.vis.remove_geometry(self.mesh)
                    self.mesh = o3d.geometry.TriangleMesh(mesh)
                    center = self.mesh.get_center()
                    self.mesh.rotate(rotation_matrix, center)
                    
                    # ビジュアライザーを更新
                    self.vis.add_geometry(self.mesh)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    
                    # フレームを画像として保存
                    global_frame_idx = axis_idx * frames_per_axis + frame_idx
                    frame_path = self.temp_frame_dir / f"frame_{global_frame_idx:05d}.png"
                    self.vis.capture_screen_image(str(frame_path), do_render=True)
                    frame_paths.append(frame_path)
                    
                    if frame_idx % 10 == 0:
                        print(f"  Frame {frame_idx + 1}/{frames_per_axis} rendered")
            
            print("Frame rendering completed")
            
        except Exception as e:
            print(f"Error during frame rendering: {e}")
            return False
        
        finally:
            # ビジュアライザーを閉じる
            if self.vis:
                self.vis.destroy_window()
        
        # 動画ファイルを作成
        if len(frame_paths) > 0:
            success = self.create_video_from_frame_files(frame_paths, str(output_path), fps)
            
            # 一時ファイルをクリーンアップ
            self.cleanup_temp_files()
            
            return success
        else:
            print("No frames rendered")
            return False
    
    def create_video_from_frame_files(self, frame_paths: List[Path], output_path: str, fps: int = 30) -> bool:
        """フレームファイルから動画ファイルを作成"""
        if not frame_paths:
            print("No frame files to create video")
            return False
        
        try:
            # 最初のフレームからサイズを取得
            first_frame = cv2.imread(str(frame_paths[0]))
            if first_frame is None:
                print(f"Error: Cannot read first frame {frame_paths[0]}")
                return False
            
            height, width = first_frame.shape[:2]
            
            # OpenCVのVideoWriterを使用
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    out.write(frame)
                else:
                    print(f"Warning: Cannot read frame {frame_path}")
                
                if i % 30 == 0:
                    print(f"  Video encoding: {i + 1}/{len(frame_paths)} frames")
            
            out.release()
            print(f"Video saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating video: {e}")
            return False
    
    def cleanup_temp_files(self):
        """一時ファイルをクリーンアップ"""
        if self.temp_frame_dir and self.temp_frame_dir.exists():
            try:
                shutil.rmtree(self.temp_frame_dir)
                print("Temporary frame files cleaned up")
            except Exception as e:
                print(f"Warning: Could not clean up temp files: {e}")

def main():
    parser = argparse.ArgumentParser(description="テクスチャ付き3Dモデル動画作成（修正版）")
    parser.add_argument("model_path", help="3Dモデルファイルのパス (OBJ推奨)")
    parser.add_argument("--frames", type=int, default=120, help="フレーム数 (デフォルト: 120)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (デフォルト: 30)")
    parser.add_argument("--axes", nargs='+', default=['y'], choices=['x', 'y', 'z'],
                       help="回転軸 (デフォルト: y)")
    parser.add_argument("--output", help="出力ディレクトリ")
    parser.add_argument("--window-size", nargs=2, type=int, default=[1280, 720],
                       help="ウィンドウサイズ [幅 高さ] (デフォルト: 1280 720)")
    parser.add_argument("--visible", action="store_true",
                       help="ビジュアライザーウィンドウを表示する")
    
    args = parser.parse_args()
    
    # 動画作成クラスを初期化
    video_creator = Open3DTexturedVideoCreator(args.output)
    
    # テクスチャ付き回転動画を作成
    success = video_creator.create_rotation_video_with_texture(
        args.model_path,
        num_frames=args.frames,
        rotation_axes=args.axes,
        window_size=tuple(args.window_size),
        fps=args.fps,
        visible=args.visible
    )
    
    if success:
        print("✓ Textured video creation completed successfully!")
        print(f"Output directory: {video_creator.output_dir}")
    else:
        print("✗ Textured video creation failed")

if __name__ == "__main__":
    main()