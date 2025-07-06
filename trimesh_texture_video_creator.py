import numpy as np
import trimesh
import cv2
import time
import shutil
import threading
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
from datetime import datetime
import subprocess
import os
from PIL import Image
import tempfile

class TrimeshTextureVideoCreator:
    """trimeshを使用したテクスチャ付き3Dモデルの動画作成クラス"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Output/trimesh_texture_video_output_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # フレーム設定
        self.temp_frame_dir = None
        self.capture_frames = []
        self.is_capturing = False
        self.capture_thread = None
        
    def load_textured_model(self, model_path: str) -> Optional[trimesh.Trimesh]:
        """テクスチャ付き3Dモデルを読み込む"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist")
            return None
        
        try:
            # trimeshでモデルを読み込み
            mesh = trimesh.load(str(model_path))
            
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                print(f"Error: No vertices found in {model_path}")
                return None
            
            print(f"Model loaded successfully:")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Has visual: {hasattr(mesh, 'visual')}")
            
            # テクスチャ情報を確認
            if hasattr(mesh.visual, 'material'):
                print(f"  Has material: True")
                if hasattr(mesh.visual.material, 'image'):
                    print(f"  Has texture image: True")
                else:
                    print(f"  Has texture image: False")
            
            if hasattr(mesh.visual, 'uv'):
                print(f"  Has UV coordinates: {mesh.visual.uv is not None}")
                if mesh.visual.uv is not None:
                    print(f"  UV coordinates: {mesh.visual.uv.shape}")
            
            # 頂点色があるかチェック
            if hasattr(mesh.visual, 'vertex_colors'):
                print(f"  Has vertex colors: {mesh.visual.vertex_colors is not None}")
                if mesh.visual.vertex_colors is not None:
                    print(f"  Vertex colors: {mesh.visual.vertex_colors.shape}")
            
            return mesh
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def get_texture_face_colors(self, mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        """テクスチャ画像からメッシュの面の色を取得"""
        try:
            # テクスチャ画像とUV座標の存在確認
            if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
                return None
                
            if not hasattr(mesh.visual.material, 'image') or mesh.visual.material.image is None:
                return None
                
            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                return None
            
            # テクスチャ画像を取得
            texture_image = mesh.visual.material.image
            if hasattr(texture_image, 'mode') and texture_image.mode == 'RGBA':
                texture_image = texture_image.convert('RGB')
            
            # テクスチャ画像をnumpy配列に変換
            texture_array = np.array(texture_image)
            tex_height, tex_width = texture_array.shape[:2]
            
            # UV座標を取得
            uv_coords = mesh.visual.uv
            
            # 面ごとのテクスチャ色を計算
            face_colors = []
            
            for face in mesh.faces:
                # 面の頂点のUV座標を取得
                face_uvs = uv_coords[face]
                
                # 各頂点のテクスチャ色を取得
                vertex_colors = []
                for uv in face_uvs:
                    # UV座標をテクスチャ画像の座標に変換
                    # UV座標は[0,1]の範囲なので、テクスチャサイズに合わせる
                    u = np.clip(uv[0], 0, 1)
                    v = np.clip(1 - uv[1], 0, 1)  # Vを反転（テクスチャ座標系）
                    
                    tex_x = int(u * (tex_width - 1))
                    tex_y = int(v * (tex_height - 1))
                    
                    # テクスチャからピクセル色を取得
                    color = texture_array[tex_y, tex_x]
                    vertex_colors.append(color)
                
                # 面の色は頂点色の平均
                face_color = np.mean(vertex_colors, axis=0)
                face_colors.append(face_color)
            
            # 正規化（0-1の範囲）
            face_colors = np.array(face_colors) / 255.0
            
            print(f"Texture face colors calculated: {len(face_colors)} faces")
            return face_colors
            
        except Exception as e:
            print(f"Error getting texture face colors: {e}")
            return None
    
    def create_single_frame(self, mesh: trimesh.Trimesh, rotation_matrix: np.ndarray, 
                          frame_size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """単一フレームを作成（matplotlib使用）"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import gc
            
            # matplotlib設定を最適化
            plt.ioff()  # インタラクティブモードを無効化
            
            # メッシュを回転
            rotated_mesh = mesh.copy()
            rotated_mesh.apply_transform(rotation_matrix)
            
            # matplotlibで3D描画
            fig = plt.figure(figsize=(frame_size[0]/100, frame_size[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # 頂点と面を取得
            vertices = rotated_mesh.vertices
            faces = rotated_mesh.faces
            
            # テクスチャ色を取得
            face_colors = self.get_texture_face_colors(rotated_mesh)
            
            if face_colors is not None:
                # テクスチャ色を使用（Poly3DCollectionで直接描画）
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                print(f"Using texture colors for {len(face_colors)} faces")
                
                # Poly3DCollectionに渡すためのポリゴン（頂点座標のリスト）を作成
                polygons = [vertices[face] for face in faces]
                
                # Poly3DCollectionを作成
                mesh_collection = Poly3DCollection(polygons, facecolors=face_colors, alpha=0.9)
                
                # Axesにメッシュを追加
                ax.add_collection3d(mesh_collection)
                
                # プロットの軸範囲を設定
                ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
                ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
                ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
                
            elif hasattr(rotated_mesh.visual, 'vertex_colors') and rotated_mesh.visual.vertex_colors is not None:
                # 頂点色を使用
                colors = rotated_mesh.visual.vertex_colors[:, :3] / 255.0
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                               triangles=faces, facecolors=colors[faces].mean(axis=1))
            else:
                # デフォルト色
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                               triangles=faces, color='lightgray', alpha=0.8)
            
            # 軸を非表示
            ax.set_axis_off()
            
            # 視点を調整
            ax.view_init(elev=20, azim=45)
            
            # 背景色を設定
            fig.patch.set_facecolor('white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # 画像をメモリに保存
            fig.canvas.draw()
            
            # RGB配列として取得（現代的なアプローチ）
            rgba_buffer = fig.canvas.buffer_rgba()
            image_array = np.asarray(rgba_buffer)
            
            # RGBデータが必要な場合、アルファチャネルを除外
            buf = image_array[:, :, :3]
            
            # メモリクリーンアップ強化
            plt.close(fig)
            plt.close('all')  # 全フィギュアを強制クリーンアップ
            
            # 明示的なガベージコレクション
            gc.collect()
            
            return buf
            
        except Exception as e:
            print(f"Error creating frame: {e}")
            # エラー時もクリーンアップ
            plt.close('all')
            gc.collect()
            return None
    
    def create_rotation_video_matplotlib(self, 
                                       model_path: str,
                                       num_frames: int = 60,
                                       rotation_axes: List[str] = ['y'],
                                       output_filename: str = None,
                                       frame_size: Tuple[int, int] = (800, 600),
                                       fps: int = 30) -> bool:
        """matplotlibを使用してテクスチャ付き回転動画作成"""
        
        # モデルを読み込み
        mesh = self.load_textured_model(model_path)
        if mesh is None:
            return False
        
        # 出力ファイル名を設定
        if output_filename is None:
            model_name = Path(model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            axes_str = "_".join(rotation_axes)
            output_filename = f"{model_name}_trimesh_texture_{axes_str}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # 一時フレームディレクトリを作成
        self.temp_frame_dir = self.output_dir / f"temp_frames_{timestamp}"
        self.temp_frame_dir.mkdir(exist_ok=True)
        
        print(f"Creating trimesh texture rotation video: {num_frames} frames, {rotation_axes} axes")
        print(f"Using matplotlib rendering with enhanced memory management")
        
        # matplotlib初期設定
        import matplotlib.pyplot as plt
        import gc
        plt.ioff()  # インタラクティブモードを無効化
        
        try:
            frame_paths = []
            frames_per_axis = num_frames // len(rotation_axes)
            
            for axis_idx, axis in enumerate(rotation_axes):
                print(f"Rendering {axis.upper()}-axis rotation...")
                
                for frame_idx in range(frames_per_axis):
                    # 回転角度を計算
                    angle_rad = (frame_idx / frames_per_axis) * 2 * np.pi
                    
                    # 回転行列を作成
                    if axis.lower() == 'x':
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])
                    elif axis.lower() == 'z':
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
                    else:  # Y軸
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
                    
                    # フレームを作成
                    frame_image = self.create_single_frame(mesh, rotation_matrix, frame_size)
                    
                    if frame_image is not None:
                        # フレームを保存
                        global_frame_idx = axis_idx * frames_per_axis + frame_idx
                        frame_path = self.temp_frame_dir / f"frame_{global_frame_idx:05d}.png"
                        
                        # RGB → BGR変換してOpenCV形式で保存
                        frame_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(frame_path), frame_bgr)
                        frame_paths.append(frame_path)
                    
                    # より頻繁なメモリクリーンアップ
                    if frame_idx % 3 == 0:
                        plt.close('all')
                        gc.collect()
                        print(f"  Frame {frame_idx + 1}/{frames_per_axis} rendered (memory cleaned)")
                    elif frame_idx % 5 == 0:
                        print(f"  Frame {frame_idx + 1}/{frames_per_axis} rendered")
            
            print("Frame rendering completed")
            
        except Exception as e:
            print(f"Error during frame rendering: {e}")
            import traceback
            traceback.print_exc()
            # エラー時の最終クリーンアップ
            plt.close('all')
            gc.collect()
            return False
        
        # 動画ファイルを作成
        if len(frame_paths) > 0:
            success = self.create_video_from_frame_files(frame_paths, str(output_path), fps)
            
            # 一時ファイルをクリーンアップ
            self.cleanup_temp_files()
            
            return success
        else:
            print("No frames rendered")
            return False
    
    def create_enhanced_trimesh_frames(self, mesh: trimesh.Trimesh, 
                                     num_frames: int = 60,
                                     rotation_axes: List[str] = ['y'],
                                     frame_size: Tuple[int, int] = (800, 600)) -> List[np.ndarray]:
        """トリメッシュフレームを作成（統合用）"""
        frames = []
        frames_per_axis = num_frames // len(rotation_axes)
        
        # matplotlib初期設定
        import matplotlib.pyplot as plt
        import gc
        plt.ioff()  # インタラクティブモードを無効化
        
        print(f"Creating enhanced trimesh frames with optimized memory management")
        
        try:
            for axis_idx, axis in enumerate(rotation_axes):
                print(f"Rendering {axis.upper()}-axis rotation frames...")
                
                for frame_idx in range(frames_per_axis):
                    # 回転角度を計算
                    angle_rad = (frame_idx / frames_per_axis) * 2 * np.pi
                    
                    # 回転行列を作成
                    if axis.lower() == 'x':
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])
                    elif axis.lower() == 'z':
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
                    else:  # Y軸
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
                    
                    # フレームを作成
                    frame_image = self.create_single_frame(mesh, rotation_matrix, frame_size)
                    
                    if frame_image is not None:
                        frames.append(frame_image)
                    else:
                        # エラーの場合は空のフレーム
                        empty_frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 240
                        frames.append(empty_frame)
                    
                    # より頻繁なメモリクリーンアップ
                    if frame_idx % 2 == 0:
                        plt.close('all')
                        gc.collect()
                        print(f"  Frame {frame_idx + 1}/{frames_per_axis} rendered (memory cleaned)")
            
            print(f"Enhanced trimesh frames creation completed: {len(frames)} frames")
            
        except Exception as e:
            print(f"Error during enhanced trimesh frames creation: {e}")
            import traceback
            traceback.print_exc()
            # エラー時の最終クリーンアップ
            plt.close('all')
            gc.collect()
        
        return frames
    
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
    parser = argparse.ArgumentParser(description="trimeshを使用したテクスチャ付き3Dモデル動画作成")
    parser.add_argument("model_path", help="3Dモデルファイルのパス (OBJ推奨)")
    parser.add_argument("--frames", type=int, default=60, help="フレーム数 (デフォルト: 60)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (デフォルト: 30)")
    parser.add_argument("--axes", nargs='+', default=['y'], choices=['x', 'y', 'z'],
                       help="回転軸 (デフォルト: y)")
    parser.add_argument("--output", help="出力ディレクトリ")
    parser.add_argument("--frame-size", nargs=2, type=int, default=[800, 600],
                       help="フレームサイズ [幅 高さ] (デフォルト: 800 600)")
    
    args = parser.parse_args()
    
    # 動画作成クラスを初期化
    video_creator = TrimeshTextureVideoCreator(args.output)
    
    # trimeshテクスチャ付き回転動画を作成
    success = video_creator.create_rotation_video_matplotlib(
        args.model_path,
        num_frames=args.frames,
        rotation_axes=args.axes,
        frame_size=tuple(args.frame_size),
        fps=args.fps
    )
    
    if success:
        print("✓ Trimesh textured video creation completed successfully!")
        print(f"Output directory: {video_creator.output_dir}")
    else:
        print("✗ Trimesh textured video creation failed")

if __name__ == "__main__":
    main()