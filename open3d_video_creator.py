import numpy as np
import open3d as o3d
import cv2
import time
import shutil
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
from datetime import datetime
import subprocess
import os

class Open3DVideoCreator:
    """Open3Dを使用したテクスチャ付き3Dモデルの動画作成クラス"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Output/open3d_video_output_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # フレーム設定
        self.temp_frame_dir = None
        
        # Open3Dビジュアライザー設定
        self.vis = None
        self.mesh = None
        self.camera_params = None
        
    def load_textured_model(self, model_path: str) -> Optional[o3d.geometry.TriangleMesh]:
        """テクスチャ付き3Dモデルを読み込む"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist")
            return None
        
        try:
            # Open3DでOBJファイルを読み込み
            mesh = o3d.io.read_triangle_mesh(str(model_path))
            
            if len(mesh.vertices) == 0:
                print(f"Error: No vertices found in {model_path}")
                return None
            
            # メッシュの基本情報を表示
            print(f"Model loaded successfully:")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Triangles: {len(mesh.triangles)}")
            print(f"  Has vertex colors: {mesh.has_vertex_colors()}")
            print(f"  Has vertex normals: {mesh.has_vertex_normals()}")
            print(f"  Has triangle normals: {mesh.has_triangle_normals()}")
            print(f"  Has textures: {mesh.has_textures()}")
            print(f"  Has triangle UVs: {mesh.has_triangle_uvs()}")
            
            # 法線が存在しない場合は計算
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
                print("  Computed vertex normals")
            
            # テクスチャ情報を確認
            if mesh.has_textures():
                print(f"  Texture images: {len(mesh.textures)}")
                for i, texture in enumerate(mesh.textures):
                    print(f"    Texture {i}: {texture.width}x{texture.height}")
            
            return mesh
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def setup_visualizer(self, window_size: Tuple[int, int] = (800, 600)) -> bool:
        """Open3Dビジュアライザーをセットアップ"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="3D Model Rotation Video",
                width=window_size[0],
                height=window_size[1],
                left=100,
                top=100
            )
            
            # レンダリングオプションを設定
            render_option = self.vis.get_render_option()
            render_option.show_coordinate_frame = False
            render_option.background_color = np.array([0.95, 0.95, 0.95])  # 薄いグレー背景
            render_option.light_on = True
            
            return True
            
        except Exception as e:
            print(f"Error setting up visualizer: {e}")
            return False
    
    def add_model_to_visualizer(self, mesh: o3d.geometry.TriangleMesh) -> bool:
        """3Dモデルをビジュアライザーに追加"""
        try:
            self.mesh = mesh
            self.vis.add_geometry(mesh)
            
            # カメラパラメータを取得・設定
            view_control = self.vis.get_view_control()
            
            # メッシュの境界を取得してカメラ位置を調整
            bbox = mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            max_extent = np.max(extent)
            
            # カメラを適切な距離に配置
            camera_distance = max_extent * 2.5
            view_control.set_lookat(center)
            view_control.set_up([0, 1, 0])  # Y軸を上方向に
            view_control.set_front([0, 0, 1])  # Z軸を前方向に
            view_control.set_zoom(0.8)
            
            # 初期レンダリング
            self.vis.poll_events()
            self.vis.update_renderer()
            
            return True
            
        except Exception as e:
            print(f"Error adding model to visualizer: {e}")
            return False
    
    def start_screen_capture(self, capture_region: Optional[Tuple[int, int, int, int]] = None):
        """画面キャプチャを開始"""
        self.capture_frames = []
        self.is_capturing = True
        
        def capture_loop():
            while self.is_capturing:
                try:
                    if capture_region:
                        # 指定領域をキャプチャ
                        screenshot = pyautogui.screenshot(region=capture_region)
                    else:
                        # 全画面をキャプチャ
                        screenshot = pyautogui.screenshot()
                    
                    # OpenCV形式に変換
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    self.capture_frames.append(frame)
                    
                    time.sleep(1/30)  # 30FPS
                    
                except Exception as e:
                    print(f"Capture error: {e}")
                    break
        
        self.capture_thread = threading.Thread(target=capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("Screen capture started")
    
    def stop_screen_capture(self):
        """画面キャプチャを停止"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join()
        print(f"Screen capture stopped. Captured {len(self.capture_frames)} frames")
    
    def create_rotation_video(self, 
                            model_path: str,
                            num_frames: int = 120,
                            rotation_axis: str = 'y',
                            output_filename: str = None,
                            window_size: Tuple[int, int] = (800, 600),
                            fps: int = 30) -> bool:
        """回転動画を作成"""
        
        # モデルを読み込み
        mesh = self.load_textured_model(model_path)
        if mesh is None:
            return False
        
        # ビジュアライザーをセットアップ
        if not self.setup_visualizer(window_size):
            return False
        
        # モデルをビジュアライザーに追加
        if not self.add_model_to_visualizer(mesh):
            return False
        
        # 出力ファイル名を設定
        if output_filename is None:
            model_name = Path(model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{model_name}_open3d_rotation_{rotation_axis}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        print(f"Creating rotation video: {num_frames} frames, {rotation_axis}-axis")
        print("Please ensure the Open3D window is visible and not occluded")
        
        # 少し待ってからキャプチャ開始
        time.sleep(2)
        
        # ウィンドウの位置とサイズを取得（手動で調整が必要な場合があります）
        # この例では全画面キャプチャを使用
        self.start_screen_capture()
        
        try:
            # 回転アニメーション
            view_control = self.vis.get_view_control()
            
            for frame_idx in range(num_frames):
                # 回転角度を計算
                angle = (frame_idx / num_frames) * 2 * np.pi
                
                # 軸に応じて回転
                if rotation_axis.lower() == 'x':
                    # X軸周りの回転
                    view_control.rotate(10, 0)
                elif rotation_axis.lower() == 'z':
                    # Z軸周りの回転
                    view_control.rotate(0, 10)
                else:  # Y軸周りの回転（デフォルト）
                    view_control.rotate(0, 0, 10)
                
                # レンダリング更新
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # フレーム間の待機時間
                time.sleep(1/fps)
                
                if frame_idx % 10 == 0:
                    print(f"Processing frame {frame_idx + 1}/{num_frames}")
            
            print("Rotation animation completed")
            
        except Exception as e:
            print(f"Error during rotation animation: {e}")
            return False
        
        finally:
            # キャプチャを停止
            time.sleep(1)  # 最後のフレームを確実にキャプチャ
            self.stop_screen_capture()
            
            # ビジュアライザーを閉じる
            self.vis.destroy_window()
        
        # 動画ファイルを作成
        if len(self.capture_frames) > 0:
            self.create_video_from_frames(self.capture_frames, str(output_path), fps)
            return True
        else:
            print("No frames captured")
            return False
    
    def create_video_from_frames(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """フレームから動画ファイルを作成"""
        if not frames:
            print("No frames to create video")
            return
        
        height, width = frames[0].shape[:2]
        
        # OpenCVのVideoWriterを使用
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved: {output_path}")
    
    def create_enhanced_rotation_video(self,
                                     model_path: str,
                                     num_rotations: int = 2,
                                     rotation_axes: List[str] = ['x', 'y', 'z'],
                                     window_size: Tuple[int, int] = (1200, 900),
                                     fps: int = 30) -> bool:
        """複数軸の回転を含む高品質動画を作成"""
        
        model_name = Path(model_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{model_name}_enhanced_rotation_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # モデルを読み込み
        mesh = self.load_textured_model(model_path)
        if mesh is None:
            return False
        
        # ビジュアライザーをセットアップ
        if not self.setup_visualizer(window_size):
            return False
        
        # モデルをビジュアライザーに追加
        if not self.add_model_to_visualizer(mesh):
            return False
        
        print(f"Creating enhanced rotation video with {len(rotation_axes)} axes")
        print("Please ensure the Open3D window is visible and not occluded")
        
        time.sleep(2)
        self.start_screen_capture()
        
        try:
            view_control = self.vis.get_view_control()
            frames_per_axis = 60 * num_rotations  # 各軸2回転分
            
            for axis in rotation_axes:
                print(f"Rotating around {axis.upper()}-axis...")
                
                for frame_idx in range(frames_per_axis):
                    # より滑らかな回転
                    rotation_step = 360 / frames_per_axis * num_rotations
                    
                    if axis.lower() == 'x':
                        view_control.rotate(rotation_step, 0)
                    elif axis.lower() == 'z':
                        view_control.rotate(0, rotation_step)
                    else:  # Y軸
                        view_control.rotate(0, 0, rotation_step)
                    
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    time.sleep(1/fps)
                
                # 軸間の小休止
                time.sleep(0.5)
            
            print("Enhanced rotation animation completed")
            
        except Exception as e:
            print(f"Error during enhanced rotation: {e}")
            return False
        
        finally:
            time.sleep(1)
            self.stop_screen_capture()
            self.vis.destroy_window()
        
        if len(self.capture_frames) > 0:
            self.create_video_from_frames(self.capture_frames, str(output_path), fps)
            return True
        else:
            print("No frames captured")
            return False

def main():
    parser = argparse.ArgumentParser(description="Open3Dを使用したテクスチャ付き3Dモデル動画作成")
    parser.add_argument("model_path", help="3Dモデルファイルのパス (OBJ推奨)")
    parser.add_argument("--frames", type=int, default=120, help="フレーム数 (デフォルト: 120)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (デフォルト: 30)")
    parser.add_argument("--axis", choices=['x', 'y', 'z'], default='y', 
                       help="回転軸 (デフォルト: y)")
    parser.add_argument("--output", help="出力ディレクトリ")
    parser.add_argument("--enhanced", action="store_true", 
                       help="複数軸の高品質動画を作成")
    parser.add_argument("--window-size", nargs=2, type=int, default=[800, 600],
                       help="ウィンドウサイズ [幅 高さ] (デフォルト: 800 600)")
    
    args = parser.parse_args()
    
    # 動画作成クラスを初期化
    video_creator = Open3DVideoCreator(args.output)
    
    if args.enhanced:
        # 高品質動画を作成
        success = video_creator.create_enhanced_rotation_video(
            args.model_path,
            num_rotations=2,
            rotation_axes=['x', 'y', 'z'],
            window_size=tuple(args.window_size),
            fps=args.fps
        )
    else:
        # 基本的な回転動画を作成
        success = video_creator.create_rotation_video(
            args.model_path,
            num_frames=args.frames,
            rotation_axis=args.axis,
            window_size=tuple(args.window_size),
            fps=args.fps
        )
    
    if success:
        print("✓ Video creation completed successfully!")
        print(f"Output directory: {video_creator.output_dir}")
    else:
        print("✗ Video creation failed")

if __name__ == "__main__":
    main()