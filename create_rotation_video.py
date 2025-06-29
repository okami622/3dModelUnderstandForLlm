import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trimesh
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image, ImageDraw, ImageFont

# モジュールのインポート
from texture_loader import TextureLoader
from x3d_loader import X3DLoader
from color_generator import ColorGenerator
from cross_section_processor import CrossSectionProcessor

class Model3DVideoCreator:
    """3Dモデルの回転動画作成クラス - Gemini分析用に最適化"""
    
    def __init__(self, output_dir: str = None, scale_reference_size: float = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Output/video_output_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # スケールリファレンス設定
        self.scale_reference_size = scale_reference_size  # ユニット単位のサイズ
        self.default_reference_ratio = 0.1  # モデルサイズに対する比率
        
        # サポートする3Dモデルの拡張子
        self.supported_formats = [
            # 一般的な3D形式
            '.ply', '.obj', '.stl', '.off', '.glb', '.gltf', '.dae', '.3mf',
            # CAD形式
            '.step', '.stp', '.iges', '.igs', '.3dxml', '.x3d',
            # その他
            '.xyz', '.dxf', '.svg'
        ]
        
        
        # 視覚化パラメータ
        self.enhanced_lighting = True
        self.use_depth_shading = True
        self.background_color = [248, 248, 248]  # 高コントラスト背景
        
        # モジュールの初期化
        self.texture_loader = TextureLoader()
        self.x3d_loader = X3DLoader()
        self.color_generator = ColorGenerator()
        self.cross_section_processor = CrossSectionProcessor()
    
        
    def load_model(self, model_path: str) -> Optional[trimesh.Trimesh]:
        """3Dモデルを読み込む"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist")
            return None
            
        if model_path.suffix.lower() not in self.supported_formats:
            print(f"Error: Unsupported format {model_path.suffix}")
            print(f"Supported formats: {self.supported_formats}")
            return None
            
        try:
            # X3D形式の特別な処理
            if model_path.suffix.lower() == '.x3d':
                mesh = self.x3d_loader.load_x3d_file(str(model_path))
                if mesh is None:
                    return None
            # CAD形式の特別な処理
            elif model_path.suffix.lower() in ['.step', '.stp', '.iges', '.igs']:
                try:
                    # CAD形式の読み込みを試行
                    mesh = trimesh.load(str(model_path))
                    print(f"Successfully loaded CAD file: {model_path.name}")
                except Exception as cad_error:
                    print(f"CAD loading error: {cad_error}")
                    print("Note: CAD formats may require additional libraries (FreeCAD, opencascade)")
                    return None
            else:
                # テクスチャローダーを使用
                mesh = self.texture_loader.load_model_with_textures(str(model_path))
            
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Error: Loaded object is not a valid mesh")
                return None
                
            print(f"Model loaded successfully: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh
            
        except Exception as e:
            print(f"Error loading model: {e}")
            if model_path.suffix.lower() in ['.step', '.stp', '.iges', '.igs']:
                print("Tip: Try converting CAD files to STL or OBJ format for better compatibility")
            return None
    
    def extract_pointcloud(self, mesh: trimesh.Trimesh, num_points: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """メッシュから点群と色情報を抽出（改善版）"""
        points, face_indices = mesh.sample(num_points, return_index=True)
        normals = mesh.face_normals[face_indices]
        
        # 色情報を改善して取得
        colors = []
        
        # メッシュに視覚情報があるかチェック
        has_vertex_colors = hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None
        has_face_colors = hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None
        has_material = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
        
        print(f"Color info - Vertex colors: {has_vertex_colors}, Face colors: {has_face_colors}, Material: {has_material}")
        
        for i, face_idx in enumerate(face_indices):
            # 面の最初の頂点を使用（高速化）
            face_vertices = mesh.faces[face_idx]
            closest_vertex = face_vertices[0]
            
            color = self.get_enhanced_point_color_with_vertex(mesh, face_idx, points[i], closest_vertex)
            colors.append(color)
        
        colors = np.array(colors)
        
        # 色の多様性をチェック
        unique_colors = len(np.unique(colors.reshape(-1, colors.shape[-1]), axis=0))
        print(f"Generated {unique_colors} unique colors from {len(colors)} points")
        
        return points, normals, colors
    
    def get_enhanced_point_color(self, mesh: trimesh.Trimesh, face_index: int, point_position: np.ndarray) -> List[int]:
        """改善された色取得（テクスチャ優先版）"""
        # メッシュにテクスチャ/マテリアル情報があるかチェック
        has_texture_info = False
        
        # 頂点色の存在と多様性をチェック
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            # 頂点色の分散を計算
            colors = mesh.visual.vertex_colors[:, :3]  # RGB成分のみ
            color_variance = np.var(colors.flatten())
            if color_variance > 50:  # 閾値を下げてより敏感に
                has_texture_info = True
                print(f"Texture detected: color variance = {color_variance:.2f}")
        
        # マテリアル情報があるかチェック
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            has_texture_info = True
            print("Material texture detected")
        
        # UV座標があるかチェック（テクスチャマッピングの証拠）
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            has_texture_info = True
            print("UV coordinates detected")
        
        # テクスチャ情報がある場合は既存の色を使用
        if has_texture_info:
            existing_color = self.get_point_color(mesh, face_index, None)
            # グレーでなければテクスチャ色を使用
            if existing_color != [150, 150, 150]:
                return existing_color
        
        # テクスチャ情報がない場合のみ位置ベース色を生成
        return self.color_generator.generate_position_based_color(point_position, mesh.bounds)
    
    def get_enhanced_point_color_with_vertex(self, mesh: trimesh.Trimesh, face_index: int, 
                                           point_position: np.ndarray, vertex_index: int) -> List[int]:
        """頂点インデックス付きの改善された色取得"""
        # メッシュにテクスチャ/マテリアル情報があるかチェック
        has_texture_info = False
        
        # 頂点色の存在と多様性をチェック
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3]  # RGB成分のみ
            color_variance = np.var(colors.flatten())
            if color_variance > 50:
                has_texture_info = True
        
        # マテリアル情報があるかチェック
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            has_texture_info = True
        
        # UV座標があるかチェック
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            has_texture_info = True
        
        # テクスチャ情報がある場合は頂点色を優先使用
        if has_texture_info:
            existing_color = self.get_point_color(mesh, face_index, vertex_index)
            if existing_color != [150, 150, 150]:
                return existing_color
        
        # テクスチャ情報がない場合のみ位置ベース色を生成
        return self.color_generator.generate_position_based_color(point_position, mesh.bounds)
    
    def create_scale_reference(self, mesh_bounds: np.ndarray) -> Dict:
        """スケールリファレンスを作成"""
        extent = np.max(mesh_bounds[1] - mesh_bounds[0])
        
        if self.scale_reference_size is not None:
            # ユーザー指定のサイズを使用
            reference_size = self.scale_reference_size
        else:
            # モデルサイズの比率から自動計算
            reference_size = extent * self.default_reference_ratio
            
        # 適切な単位を選択
        if reference_size >= 1.0:
            unit = "unit"
            display_size = reference_size
        elif reference_size >= 0.01:
            unit = "cm"
            display_size = reference_size * 100
        else:
            unit = "mm"
            display_size = reference_size * 1000
            
        return {
            'size': reference_size,
            'display_size': display_size,
            'unit': unit,
            'extent_ratio': reference_size / extent
        }
    
    def add_scale_reference_to_image(self, img: np.ndarray, scale_info: Dict, 
                                   position: str = "bottom_right") -> np.ndarray:
        """画像にスケールリファレンスを追加"""
        from PIL import Image, ImageDraw, ImageFont
        
        img_uint8 = img.astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            
        # スケールバーのサイズ（ピクセル）
        bar_length = 50  # 固定長
        bar_height = 3
        
        # 位置を計算
        margin = 15
        if position == "bottom_right":
            x = img.shape[1] - bar_length - margin
            y = img.shape[0] - 30 - margin
        elif position == "bottom_left":
            x = margin
            y = img.shape[0] - 30 - margin
        else:  # top_right
            x = img.shape[1] - bar_length - margin
            y = margin + 20
            
        # スケールバーを描画
        draw.rectangle([x, y, x + bar_length, y + bar_height], 
                      fill=(0, 0, 0), outline=(0, 0, 0))
        
        # テキストを描画
        text = f"{scale_info['display_size']:.1f} {scale_info['unit']}"
        draw.text((x + bar_length // 2, y + bar_height + 2), text, 
                 fill=(0, 0, 0), font=font, anchor="ma")
        
        return np.array(pil_img)
    
    def add_text_to_image(self, img: np.ndarray, projection_type: str, view_name: str, 
                         frame_number: int = None, scale_info: Dict = None) -> np.ndarray:
        """画像にテキストを追加"""
        # NumPy配列をPIL Imageに変換（データ型を修正）
        img_uint8 = img.astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            # フォントサイズを調整
            font_size = 20
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # テキストを描画
        text_lines = [
            f"Projection: {projection_type}",
            f"View: {view_name}"
        ]
        
        if frame_number is not None:
            text_lines.append(f"Frame: {frame_number}")
        
        y_offset = 10
        for line in text_lines:
            draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += 25
        
        # PIL ImageをNumPy配列に戻してスケールリファレンスを追加
        img_with_text = np.array(pil_img)
        
        if scale_info is not None:
            img_with_text = self.add_scale_reference_to_image(img_with_text, scale_info)
        
        return img_with_text
    
    def create_wireframe_rotation_frames(self, mesh: trimesh.Trimesh, 
                                       num_frames: int = 60,
                                       image_size: Tuple[int, int] = (800, 600),
                                       rotation_axis: str = 'y',
                                       with_scale: bool = True) -> List[np.ndarray]:
        """ワイヤーフレームの回転フレームを作成"""
        frames = []
        
        # メッシュの境界を取得
        bounds = mesh.bounds
        center = mesh.centroid
        extent = np.max(bounds[1] - bounds[0])
        distance = extent * 2.5
        
        # スケールリファレンス情報を作成
        scale_info = self.create_scale_reference(bounds) if with_scale else None
        
        for frame in range(num_frames):
            # 回転角度を計算（360度を num_frames で分割）
            angle = (frame / num_frames) * 2 * np.pi
            
            # 回転行列を作成
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            if rotation_axis == 'x':
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, cos_a, -sin_a],
                    [0, sin_a, cos_a]
                ])
            elif rotation_axis == 'z':
                rotation_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
            else:  # 'y'
                rotation_matrix = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
            
            # 統一された一方向回転（点群と同じロジック）
            if rotation_axis == 'x':
                # X軸周り: YZ平面で一方向回転
                camera_pos = center + np.array([0, np.cos(angle), np.sin(angle)]) * distance
            elif rotation_axis == 'y':
                # Y軸周り: XZ平面で一方向回転
                camera_pos = center + np.array([np.sin(angle), 0, np.cos(angle)]) * distance
            else:  # 'z'
                # Z軸周り: XY平面で一方向回転
                camera_pos = center + np.array([np.cos(angle), np.sin(angle), 0]) * distance
            
            # 視線ベクトル
            view_vector = center - camera_pos
            view_vector = view_vector / np.linalg.norm(view_vector)
            
            # 上方向ベクトルを動的に設定（視線ベクトルと平行にならないように）
            if abs(view_vector[1]) > 0.9:  # Y軸とほぼ平行の場合
                up_vector = np.array([0, 0, 1])  # Z軸を上方向に
            else:
                up_vector = np.array([0, 1, 0])  # Y軸を上方向に
            
            # 右方向ベクトル
            right_vector = np.cross(view_vector, up_vector)
            right_norm = np.linalg.norm(right_vector)
            if right_norm > 1e-6:
                right_vector = right_vector / right_norm
            else:
                # フォールバック
                right_vector = np.array([1, 0, 0])
            
            # 上方向ベクトルを再計算
            up_vector = np.cross(right_vector, view_vector)
            up_norm = np.linalg.norm(up_vector)
            if up_norm > 1e-6:
                up_vector = up_vector / up_norm
            else:
                # フォールバック
                up_vector = np.array([0, 1, 0])
            
            # ワイヤーフレーム画像を作成
            img = self._render_wireframe(mesh, center, camera_pos, view_vector, 
                                       right_vector, up_vector, image_size)
            
            # テキストとスケールを追加
            img = self.add_text_to_image(img, "Wireframe", f"Rotation {rotation_axis.upper()}", frame, scale_info)
            frames.append(img)
        
        return frames
    
    def create_pointcloud_rotation_frames(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
                                        num_frames: int = 60,
                                        image_size: Tuple[int, int] = (800, 600),
                                        rotation_axis: str = 'y',
                                        with_scale: bool = True) -> List[np.ndarray]:
        """点群の回転フレームを作成"""
        frames = []
        
        # 点群の境界を取得
        center = points.mean(axis=0)
        extent = np.max(points.max(axis=0) - points.min(axis=0))
        distance = extent * 3.0  # 距離を増加してより良い視点を確保
        
        # スケールリファレンス情報を作成
        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        scale_info = self.create_scale_reference(bounds) if with_scale else None
        
        for frame in range(num_frames):
            # 回転角度を計算（完全一方向回転）
            angle = (frame / num_frames) * 2 * np.pi
            if frame == 0 or frame == num_frames // 2 or frame == num_frames - 1:
                pass  # ログを削除
            
            # 回転行列を作成（カメラを一方向に回転させる）
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            if rotation_axis == 'x':
                # X軸周りの一方向回転
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, cos_a, -sin_a],
                    [0, sin_a, cos_a]
                ])
            elif rotation_axis == 'z':
                # Z軸周りの一方向回転
                rotation_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
            else:  # 'y'
                # Y軸周りの一方向回転
                rotation_matrix = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
            
            # 統一された一方向回転（右手座標系）
            if rotation_axis == 'x':
                # X軸周り: YZ平面で一方向回転
                camera_pos = center + np.array([0, np.cos(angle), np.sin(angle)]) * distance
            elif rotation_axis == 'y':
                # Y軸周り: XZ平面で一方向回転
                camera_pos = center + np.array([np.sin(angle), 0, np.cos(angle)]) * distance
            else:  # 'z'
                # Z軸周り: XY平面で一方向回転
                camera_pos = center + np.array([np.cos(angle), np.sin(angle), 0]) * distance
            
            # 視線ベクトル（カメラからオブジェクトへ）
            direction = center - camera_pos
            direction = direction / np.linalg.norm(direction)
            
            
            # 点群画像を作成
            img = self._render_pointcloud_perspective(points, normals, colors, center, camera_pos, 
                                                   direction, image_size)
            
            # テキストとスケールを追加
            img = self.add_text_to_image(img, "Point Cloud", f"Rotation {rotation_axis.upper()}", frame, scale_info)
            frames.append(img)
        
        return frames
    
    def _render_wireframe(self, mesh: trimesh.Trimesh, center: np.ndarray, 
                         camera_pos: np.ndarray, view_vector: np.ndarray,
                         right_vector: np.ndarray, up_vector: np.ndarray,
                         image_size: Tuple[int, int]) -> np.ndarray:
        """ワイヤーフレームをレンダリング"""
        # 透視投影パラメータ
        fov = 45
        aspect_ratio = image_size[0] / image_size[1]
        
        # 背景色を改善
        img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)
        
        # 深度情報を計算
        all_depths = []
        for vertex in mesh.vertices:
            depth = np.linalg.norm(vertex - camera_pos)
            all_depths.append(depth)
        max_depth = max(all_depths) if all_depths else 1.0
        
        # エッジを効率的に抽出（重複排除）
        edges = mesh.edges_unique
        
        # LOD（Level of Detail）を適用 - 大きなメッシュの場合はエッジを間引く
        if len(edges) > 50000:  # 5万エッジより多い場合は間引く
            step = max(1, len(edges) // 30000)  # 最大3万エッジに制限
            edges = edges[::step]
        else:
            pass  # ログを削除
        
        # 各エッジを描画
        for edge in edges:
            start_vertex = mesh.vertices[edge[0]]
            end_vertex = mesh.vertices[edge[1]]
            
            # 両端点を透視投影
            start_2d = self._project_point_perspective(start_vertex, camera_pos, 
                                                        view_vector, right_vector, up_vector,
                                                        fov, aspect_ratio, image_size)
            end_2d = self._project_point_perspective(end_vertex, camera_pos,
                                                      view_vector, right_vector, up_vector,
                                                      fov, aspect_ratio, image_size)
            
            # 線を描画（深度シェーディング付き）
            if start_2d is not None and end_2d is not None:
                # エッジの中点での深度を計算
                mid_vertex = (start_vertex + end_vertex) / 2
                depth = np.linalg.norm(mid_vertex - camera_pos)
                line_color = self.color_generator.apply_depth_shading([80, 80, 80], depth, max_depth)
                self._draw_line(img, start_2d, end_2d, line_color)
        
        return img
    
    def get_point_color(self, mesh: trimesh.Trimesh, face_index: int, point_index: int = None) -> List[int]:
        """点の色をメッシュから取得（テクスチャ優先版）"""
        # デフォルト色（グレー）
        default_color = [150, 150, 150]
        
        try:
            # 頂点の色情報を優先してチェック（テクスチャ情報）
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                if point_index is not None and point_index < len(mesh.visual.vertex_colors):
                    vertex_color = mesh.visual.vertex_colors[point_index]
                    # RGBA値の処理
                    if len(vertex_color) >= 3:
                        if np.max(vertex_color[:3]) <= 1.0:
                            color_rgb = [int(vertex_color[0] * 255), int(vertex_color[1] * 255), int(vertex_color[2] * 255)]
                        else:
                            color_rgb = [int(vertex_color[0]), int(vertex_color[1]), int(vertex_color[2])]
                        
                        # 有効な色かチェック（黒や白以外）
                        if not (color_rgb == [0, 0, 0] or color_rgb == [255, 255, 255]):
                            return color_rgb
                
                # 面の頂点から平均色を計算（テクスチャ色の保持）
                if face_index < len(mesh.faces):
                    face_vertices = mesh.faces[face_index]
                    valid_colors = []
                    
                    for vertex_idx in face_vertices:
                        if vertex_idx < len(mesh.visual.vertex_colors):
                            vertex_color = mesh.visual.vertex_colors[vertex_idx]
                            if len(vertex_color) >= 3:
                                if np.max(vertex_color[:3]) <= 1.0:
                                    color = [int(vertex_color[0] * 255), int(vertex_color[1] * 255), int(vertex_color[2] * 255)]
                                else:
                                    color = [int(vertex_color[0]), int(vertex_color[1]), int(vertex_color[2])]
                                
                                # 有効な色のみ追加
                                if not (color == [0, 0, 0] or color == [255, 255, 255]):
                                    valid_colors.append(color)
                    
                    if valid_colors:
                        # 平均色を計算
                        avg_color = np.mean(valid_colors, axis=0)
                        return [int(avg_color[0]), int(avg_color[1]), int(avg_color[2])]
            
            # 面の色情報をチェック
            if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                if face_index < len(mesh.visual.face_colors):
                    face_color = mesh.visual.face_colors[face_index]
                    if np.max(face_color[:3]) <= 1.0:
                        return [int(face_color[0] * 255), int(face_color[1] * 255), int(face_color[2] * 255)]
                    else:
                        return [int(face_color[0]), int(face_color[1]), int(face_color[2])]
            
            # マテリアルの色情報をチェック
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                if hasattr(mesh.visual.material, 'diffuse'):
                    diffuse = mesh.visual.material.diffuse
                    if diffuse is not None and len(diffuse) >= 3:
                        if np.max(diffuse[:3]) <= 1.0:
                            return [int(diffuse[0] * 255), int(diffuse[1] * 255), int(diffuse[2] * 255)]
                        else:
                            return [int(diffuse[0]), int(diffuse[1]), int(diffuse[2])]
            
            return default_color
            
        except Exception as e:
            print(f"Color extraction error: {e}")
            return default_color
    
    def _render_pointcloud_perspective(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
                                     center: np.ndarray, camera_pos: np.ndarray,
                                     direction: np.ndarray, 
                                     image_size: Tuple[int, int]) -> np.ndarray:
        """点群を簡単な平行投影でレンダリング（動作を確実にするため）"""
        # 画像を作成（改善された背景色）
        img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * np.array(self.background_color)
        
        # 深度情報を計算
        depths = np.linalg.norm(points - camera_pos, axis=1)
        max_depth = np.max(depths) if len(depths) > 0 else 1.0
        
        # 上方向ベクトルを設定
        if abs(direction[1]) > 0.9:  # 上面または下面ビュー
            up_vector = np.array([0, 0, 1])
        else:
            up_vector = np.array([0, 1, 0])
        
        # 右方向ベクトルを計算
        right_vector = np.cross(direction, up_vector)
        if np.linalg.norm(right_vector) > 0:
            right_vector = right_vector / np.linalg.norm(right_vector)
        else:
            right_vector = np.array([1, 0, 0])
        
        # 上方向ベクトルを再計算
        up_vector = np.cross(right_vector, direction)
        if np.linalg.norm(up_vector) > 0:
            up_vector = up_vector / np.linalg.norm(up_vector)
        else:
            up_vector = np.array([0, 1, 0])
        
        # 点群を中心に移動
        centered_points = points - center
        
        # 2D投影を作成（平行投影）
        projected_x = np.dot(centered_points, right_vector)
        projected_y = np.dot(centered_points, up_vector)
        
        # 投影範囲を計算
        x_min, x_max = projected_x.min(), projected_x.max()
        y_min, y_max = projected_y.min(), projected_y.max()
        
        # 画像のマージンを追加
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin
        
        # 各点を描画
        for i, (point_x, point_y) in enumerate(zip(projected_x, projected_y)):
            # 画像座標に変換
            if x_max != x_min and y_max != y_min:  # ゼロ除算を防ぐ
                pixel_x = int((point_x - x_min) / (x_max - x_min) * (image_size[0] - 1))
                pixel_y = int((point_y - y_min) / (y_max - y_min) * (image_size[1] - 1))
                pixel_y = image_size[1] - 1 - pixel_y  # Y軸を反転
                
                # 背面カリングチェック
                visible = True
                if normals is not None and i < len(normals):
                    normal = normals[i]
                    dot_product = np.dot(normal, direction)
                    # 背面の点は表示しない
                    if dot_product < 0:
                        visible = False
                
                # モデルの色を使用（深度シェーディング付き）
                if i < len(colors):
                    base_color = [int(colors[i][0]), int(colors[i][1]), int(colors[i][2])]
                else:
                    base_color = [150, 150, 150]
                
                # 深度シェーディングを適用
                depth = depths[i] if i < len(depths) else max_depth
                point_color = self.color_generator.apply_depth_shading(base_color, depth, max_depth)
                
                # 見える点のみ描画（より大きなサイズで）
                if visible and 0 <= pixel_x < image_size[0] and 0 <= pixel_y < image_size[1]:
                    # 可変サイズの点を描画
                    point_size = 2
                    for dx in range(-point_size, point_size + 1):
                        for dy in range(-point_size, point_size + 1):
                            px, py = pixel_x + dx, pixel_y + dy
                            if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                                distance = (dx**2 + dy**2)**0.5
                                if distance <= point_size:
                                    img[py, px] = point_color
        
        return img
    
    def create_cross_section_frames(self, mesh: trimesh.Trimesh, points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
                                   num_frames: int = 60,
                                   image_size: Tuple[int, int] = (800, 600),
                                   section_axis: str = 'z') -> List[np.ndarray]:
        """真の幾何学的断面図フレームを作成"""
        frames = []
        
        # 点群の境界を取得
        center = points.mean(axis=0)
        
        # 断面軸を設定
        if section_axis == 'x':
            axis_index = 0
            section_min = points[:, 0].min()
            section_max = points[:, 0].max()
        elif section_axis == 'y':
            axis_index = 1
            section_min = points[:, 1].min()
            section_max = points[:, 1].max()
        else:  # 'z'
            axis_index = 2
            section_min = points[:, 2].min()
            section_max = points[:, 2].max()
        
        # 断面の範囲を設定（端から端まで完全に）
        range_size = section_max - section_min
        start_pos = section_min + range_size * 0.05  # 端から5%マージン
        end_pos = section_max - range_size * 0.05    # 端まで5%マージン
        
        for frame in range(num_frames):
            # 断面位置を計算
            t = frame / (num_frames - 1) if num_frames > 1 else 0
            section_pos = start_pos + t * (end_pos - start_pos)
            
            # 真の幾何学的断面を計算
            cross_section_points, cross_section_colors = self._compute_geometric_cross_section(
                mesh, section_pos, section_axis)
            
            # 断面情報を追加
            section_info = {
                'axis': section_axis,
                'position': section_pos,
                'range_min': section_min,
                'range_max': section_max,
                'point_count': len(cross_section_points)
            }
            
            if len(cross_section_points) == 0:
                # 点がない場合は空の画像
                img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240
                progress = (frame / (num_frames - 1)) * 100 if num_frames > 1 else 0
                img = self.add_enhanced_cross_section_text(img, section_axis, section_pos, 
                                                         section_min, section_max, progress, frame)
                # 空の断面を示すメッセージを追加
                img = self._add_no_section_message(img)
                frames.append(img)
                continue
            
            # 真の断面を描画
            img = self._render_geometric_cross_section(cross_section_points, cross_section_colors,
                                                     section_axis, image_size, section_info)
            
            # テキストを追加（断面情報を詳細に）
            progress = (frame / (num_frames - 1)) * 100 if num_frames > 1 else 0
            img = self.add_enhanced_cross_section_text(img, section_axis, section_pos, 
                                                     section_min, section_max, progress, frame)
            frames.append(img)
        
        return frames
    
    def _compute_geometric_cross_section(self, mesh: trimesh.Trimesh, section_pos: float, 
                                       section_axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """真の幾何学的断面を計算"""
        try:
            # 断面平面を定義
            if section_axis == 'x':
                plane_normal = np.array([1, 0, 0])
                plane_origin = np.array([section_pos, 0, 0])
            elif section_axis == 'y':
                plane_normal = np.array([0, 1, 0])
                plane_origin = np.array([0, section_pos, 0])
            else:  # 'z'
                plane_normal = np.array([0, 0, 1])
                plane_origin = np.array([0, 0, section_pos])
            
            # メッシュと平面の交線を計算
            try:
                slice_result = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
                
                if slice_result is not None and hasattr(slice_result, 'vertices'):
                    # 交線の頂点を取得
                    section_points = slice_result.vertices
                    
                    # 断面の色を決定（構造を示すための適切な色）
                    section_colors = self._generate_cross_section_colors(section_points, section_axis)
                    
                    pass  # 詳細ログを削除
                    return section_points, section_colors
                    
            except Exception as slice_error:
                print(f"Mesh slice failed: {slice_error}")
            
            # フォールバック: エッジと平面の交点を手動計算
            intersection_points = []
            
            for edge in mesh.edges_unique:
                v1, v2 = mesh.vertices[edge]
                
                # エッジが平面を横切るかチェック
                d1 = np.dot(v1 - plane_origin, plane_normal)
                d2 = np.dot(v2 - plane_origin, plane_normal)
                
                # エッジが平面を横切る場合
                if d1 * d2 < 0:  # 符号が異なる
                    # 交点を計算
                    t = d1 / (d1 - d2)
                    intersection = v1 + t * (v2 - v1)
                    intersection_points.append(intersection)
            
            if intersection_points:
                section_points = np.array(intersection_points)
                section_colors = self._generate_cross_section_colors(section_points, section_axis)
                print(f"Manual cross-section: {len(section_points)} intersection points")
                return section_points, section_colors
            
        except Exception as e:
            print(f"Cross-section computation failed: {e}")
        
        # すべて失敗した場合は空を返す
        return np.array([]), np.array([])
    
    def _generate_cross_section_colors(self, points: np.ndarray, section_axis: str) -> np.ndarray:
        """断面用の適切な色を生成"""
        if len(points) == 0:
            return np.array([])
        
        # 断面軸に応じた基本色を設定
        base_colors = {
            'x': [220, 100, 100],  # 赤系
            'y': [100, 220, 100],  # 緑系
            'z': [100, 100, 220]   # 青系
        }
        base_color = base_colors.get(section_axis, [150, 150, 150])
        
        # 断面内での位置に基づいて色の濃淡を変える
        colors = []
        
        # 中心からの距離を計算
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        for i, distance in enumerate(distances):
            # 中心から遠いほど濃く、近いほど薄く
            intensity = 0.3 + 0.7 * (distance / max_distance) if max_distance > 0 else 0.5
            color = [int(c * intensity) for c in base_color]
            colors.append(color)
        
        return np.array(colors)
    
    def _render_geometric_cross_section(self, points: np.ndarray, colors: np.ndarray,
                                      section_axis: str, image_size: Tuple[int, int], 
                                      section_info: Dict) -> np.ndarray:
        """幾何学的断面を正確に描画"""
        # 背景を白に設定
        img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
        
        if len(points) == 0:
            return img
        
        # 断面軸に応じて投影軸を選択
        if section_axis == 'x':
            # YZ平面に投影
            proj_x = points[:, 1]  # Y軸
            proj_y = points[:, 2]  # Z軸
        elif section_axis == 'y':
            # XZ平面に投影
            proj_x = points[:, 0]  # X軸
            proj_y = points[:, 2]  # Z軸
        else:  # 'z'
            # XY平面に投影
            proj_x = points[:, 0]  # X軸
            proj_y = points[:, 1]  # Y軸
        
        if len(proj_x) == 0:
            return img
        
        # 投影範囲を計算
        x_min, x_max = proj_x.min(), proj_x.max()
        y_min, y_max = proj_y.min(), proj_y.max()
        
        # アスペクト比を保持してマージンを追加
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)
        
        # 正方形に調整（円形断面を正しく表示するため）
        max_range = max(x_range, y_range)
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        
        margin = 0.15
        x_min = x_center - max_range * (0.5 + margin)
        x_max = x_center + max_range * (0.5 + margin)
        y_min = y_center - max_range * (0.5 + margin)
        y_max = y_center + max_range * (0.5 + margin)
        
        # 点を並べ替えて線を描画（輪郭線として）
        if len(points) > 2:
            # 点を角度順に並べ替え
            center_2d = np.array([proj_x.mean(), proj_y.mean()])
            angles = np.arctan2(proj_y - center_2d[1], proj_x - center_2d[0])
            sorted_indices = np.argsort(angles)
            
            # 並べ替えた順序で線を描画
            prev_pixel = None
            for i in sorted_indices:
                point_x, point_y = proj_x[i], proj_y[i]
                
                # 画像座標に変換
                pixel_x = int((point_x - x_min) / (x_max - x_min) * (image_size[0] - 1))
                pixel_y = int((point_y - y_min) / (y_max - y_min) * (image_size[1] - 1))
                pixel_y = image_size[1] - 1 - pixel_y  # Y軸を反転
                
                if 0 <= pixel_x < image_size[0] and 0 <= pixel_y < image_size[1]:
                    # 線の色
                    if i < len(colors):
                        line_color = colors[i].tolist()
                    else:
                        line_color = [150, 150, 150]
                    
                    # 線を描画
                    if prev_pixel is not None:
                        cv2.line(img, prev_pixel, (pixel_x, pixel_y), line_color, 2)
                    
                    # 点も描画
                    cv2.circle(img, (pixel_x, pixel_y), 3, line_color, -1)
                    
                    prev_pixel = (pixel_x, pixel_y)
            
            # 最後の点と最初の点を繋ぐ
            if prev_pixel is not None and len(sorted_indices) > 0:
                first_idx = sorted_indices[0]
                first_x = int((proj_x[first_idx] - x_min) / (x_max - x_min) * (image_size[0] - 1))
                first_y = int((proj_y[first_idx] - y_min) / (y_max - y_min) * (image_size[1] - 1))
                first_y = image_size[1] - 1 - first_y
                
                if 0 <= first_x < image_size[0] and 0 <= first_y < image_size[1]:
                    first_color = colors[first_idx].tolist() if first_idx < len(colors) else [150, 150, 150]
                    cv2.line(img, prev_pixel, (first_x, first_y), first_color, 2)
        
        return img
    
    def _render_enhanced_cross_section(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
                                     section_axis: str, image_size: Tuple[int, int], 
                                     section_info: Dict) -> np.ndarray:
        """改善された断面レンダリング（色付きで詳細表示）"""
        # 背景を淡いグレーに設定
        img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240
        
        if len(points) == 0:
            return img
        
        # 断面軸に応じて投影軸を選択
        if section_axis == 'x':
            # YZ平面に投影
            proj_x = points[:, 1]  # Y軸
            proj_y = points[:, 2]  # Z軸
        elif section_axis == 'y':
            # XZ平面に投影
            proj_x = points[:, 0]  # X軸
            proj_y = points[:, 2]  # Z軸
        else:  # 'z'
            # XY平面に投影
            proj_x = points[:, 0]  # X軸
            proj_y = points[:, 1]  # Y軸
        
        # 投影範囲を計算
        x_min, x_max = proj_x.min(), proj_x.max()
        y_min, y_max = proj_y.min(), proj_y.max()
        
        # マージンを追加
        margin = 0.1
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin
        
        # 各点を描画（色付きで大きなサイズ）
        for i, (point_x, point_y) in enumerate(zip(proj_x, proj_y)):
            # 画像座標に変換
            pixel_x = int((point_x - x_min) / (x_max - x_min) * (image_size[0] - 1))
            pixel_y = int((point_y - y_min) / (y_max - y_min) * (image_size[1] - 1))
            pixel_y = image_size[1] - 1 - pixel_y  # Y軸を反転
            
            # 色を決定（改善された色情報を使用）
            if i < len(colors):
                point_color = [int(colors[i][0]), int(colors[i][1]), int(colors[i][2])]
            else:
                # フォールバック色（断面の深度に基づく）
                depth_ratio = i / len(points) if len(points) > 0 else 0
                point_color = [
                    int(255 * (1 - depth_ratio * 0.5)),  # 赤成分
                    int(255 * depth_ratio),               # 緑成分  
                    int(150)                              # 青成分
                ]
            
            # より大きな点を描画（断面の詳細を強調）
            point_size = 4
            if 0 <= pixel_x < image_size[0] and 0 <= pixel_y < image_size[1]:
                for dx in range(-point_size, point_size + 1):
                    for dy in range(-point_size, point_size + 1):
                        px, py = pixel_x + dx, pixel_y + dy
                        if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                            distance = (dx**2 + dy**2)**0.5
                            if distance <= point_size:
                                # 距離に基づく色の減衰
                                intensity = max(0.3, 1.0 - distance / point_size)
                                final_color = [int(c * intensity) for c in point_color]
                                img[py, px] = final_color
        
        # 断面の輪郭を描画（改善版）
        self._draw_enhanced_cross_section_outline(img, proj_x, proj_y, x_min, x_max, y_min, y_max, 
                                                image_size, section_info)
        
        return img
    
    def _render_cross_section(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
                             section_axis: str, image_size: Tuple[int, int]) -> np.ndarray:
        """断面を改善されたレンダリングで表示"""
        # 背景を淡いグレーに設定
        img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240
        
        if len(points) == 0:
            return img
        
        # 断面軸に応じて投影軸を選択
        if section_axis == 'x':
            # YZ平面に投影
            proj_x = points[:, 1]  # Y軸
            proj_y = points[:, 2]  # Z軸
            axis_values = points[:, 0]  # X軸の値
        elif section_axis == 'y':
            # XZ平面に投影
            proj_x = points[:, 0]  # X軸
            proj_y = points[:, 2]  # Z軸
            axis_values = points[:, 1]  # Y軸の値
        else:  # 'z'
            # XY平面に投影
            proj_x = points[:, 0]  # X軸
            proj_y = points[:, 1]  # Y軸
            axis_values = points[:, 2]  # Z軸の値
        
        # 投影範囲を計算
        x_min, x_max = proj_x.min(), proj_x.max()
        y_min, y_max = proj_y.min(), proj_y.max()
        
        # マージンを追加
        margin = 0.1
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin
        
        # 各点を描画
        for i, (point_x, point_y) in enumerate(zip(proj_x, proj_y)):
            # 画像座標に変換
            if x_max != x_min and y_max != y_min:
                pixel_x = int((point_x - x_min) / (x_max - x_min) * (image_size[0] - 1))
                pixel_y = int((point_y - y_min) / (y_max - y_min) * (image_size[1] - 1))
                pixel_y = image_size[1] - 1 - pixel_y  # Y軸を反転
                
                # モデルの色を使用
                if i < len(colors):
                    point_color = [int(colors[i][0]), int(colors[i][1]), int(colors[i][2])]
                else:
                    point_color = [150, 150, 150]
                point_size = 3  # デフォルトサイズ
                
                # 点を描画
                if 0 <= pixel_x < image_size[0] and 0 <= pixel_y < image_size[1]:
                    # 可変サイズの点を描画
                    for dx in range(-point_size, point_size + 1):
                        for dy in range(-point_size, point_size + 1):
                            px, py = pixel_x + dx, pixel_y + dy
                            if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                                # 中心からの距離による表示制御
                                distance = ((dx**2 + dy**2)**0.5) / point_size
                                if distance <= 1.0:
                                    img[py, px] = point_color
        
        # 断面の輪郭線を描画（コンベックスハル）
        self._draw_cross_section_outline(img, proj_x, proj_y, x_min, x_max, y_min, y_max, image_size)
        
        return img
    
    def _draw_enhanced_cross_section_outline(self, img: np.ndarray, proj_x: np.ndarray, proj_y: np.ndarray,
                                           x_min: float, x_max: float, y_min: float, y_max: float,
                                           image_size: Tuple[int, int], section_info: Dict):
        """断面の改善された輪郭線を描画"""
        try:
            from scipy.spatial import ConvexHull
            
            # 2D投影点を結合
            points_2d = np.column_stack([proj_x, proj_y])
            
            if len(points_2d) < 3:
                return
            
            # コンベックスハルを計算
            hull = ConvexHull(points_2d)
            
            # ハルの頂点を画像座標に変換
            hull_points_pixel = []
            for vertex in hull.vertices:
                point_x, point_y = points_2d[vertex]
                pixel_x = int((point_x - x_min) / (x_max - x_min) * (image_size[0] - 1))
                pixel_y = int((point_y - y_min) / (y_max - y_min) * (image_size[1] - 1))
                pixel_y = image_size[1] - 1 - pixel_y
                hull_points_pixel.append((pixel_x, pixel_y))
            
            # 断面の厚みに応じて線の太さを調整
            line_thickness = max(1, int(4 * section_info['thickness'] / 0.01))
            
            # 輪郭線を描画（色は断面軸に応じて変更）
            outline_colors = {
                'x': (220, 50, 50),   # 赤系
                'y': (50, 220, 50),   # 緑系  
                'z': (50, 50, 220)    # 青系
            }
            outline_color = outline_colors.get(section_info['axis'], (100, 100, 100))
            
            for i in range(len(hull_points_pixel)):
                start = hull_points_pixel[i]
                end = hull_points_pixel[(i + 1) % len(hull_points_pixel)]
                cv2.line(img, start, end, outline_color, line_thickness)
                
        except ImportError:
            # scipyがない場合は簡単な輪郭を描画
            self._draw_simple_boundary(img, proj_x, proj_y, x_min, x_max, y_min, y_max, image_size)
        except Exception:
            # エラーの場合は簡単な輪郭を描画
            self._draw_simple_boundary(img, proj_x, proj_y, x_min, x_max, y_min, y_max, image_size)
    
    def _draw_cross_section_outline(self, img: np.ndarray, proj_x: np.ndarray, proj_y: np.ndarray,
                                   x_min: float, x_max: float, y_min: float, y_max: float,
                                   image_size: Tuple[int, int]):
        """断面の輪郭線を描画"""
        try:
            from scipy.spatial import ConvexHull
            
            # 2D投影点を結合
            points_2d = np.column_stack([proj_x, proj_y])
            
            if len(points_2d) < 3:
                return
            
            # コンベックスハルを計算
            hull = ConvexHull(points_2d)
            
            # ハルの頂点を画像座標に変換
            hull_points_pixel = []
            for vertex in hull.vertices:
                point_x, point_y = points_2d[vertex]
                pixel_x = int((point_x - x_min) / (x_max - x_min) * (image_size[0] - 1))
                pixel_y = int((point_y - y_min) / (y_max - y_min) * (image_size[1] - 1))
                pixel_y = image_size[1] - 1 - pixel_y
                hull_points_pixel.append((pixel_x, pixel_y))
            
            # 輪郭線を描画
            for i in range(len(hull_points_pixel)):
                start = hull_points_pixel[i]
                end = hull_points_pixel[(i + 1) % len(hull_points_pixel)]
                cv2.line(img, start, end, (50, 50, 50), 2)  # 濃いグレーの輪郭線
                
        except ImportError:
            # scipyがない場合は簡単な料郭を描画
            self._draw_simple_boundary(img, proj_x, proj_y, x_min, x_max, y_min, y_max, image_size)
        except Exception:
            # エラーの場合は簡単な料郭を描画
            self._draw_simple_boundary(img, proj_x, proj_y, x_min, x_max, y_min, y_max, image_size)
    
    def _draw_simple_boundary(self, img: np.ndarray, proj_x: np.ndarray, proj_y: np.ndarray,
                             x_min: float, x_max: float, y_min: float, y_max: float,
                             image_size: Tuple[int, int]):
        """簡単な境界線を描画"""
        # 最外部の点を探して矩形を描画
        actual_x_min, actual_x_max = proj_x.min(), proj_x.max()
        actual_y_min, actual_y_max = proj_y.min(), proj_y.max()
        
        # 画像座標に変換
        x1 = int((actual_x_min - x_min) / (x_max - x_min) * (image_size[0] - 1))
        x2 = int((actual_x_max - x_min) / (x_max - x_min) * (image_size[0] - 1))
        y1 = int((actual_y_min - y_min) / (y_max - y_min) * (image_size[1] - 1))
        y2 = int((actual_y_max - y_min) / (y_max - y_min) * (image_size[1] - 1))
        
        y1 = image_size[1] - 1 - y1
        y2 = image_size[1] - 1 - y2
        
        # 矩形を描画
        cv2.rectangle(img, (x1, min(y1, y2)), (x2, max(y1, y2)), (100, 100, 100), 2)
    
    def add_enhanced_cross_section_text(self, img: np.ndarray, section_axis: str, 
                                       section_pos: float, section_min: float, section_max: float,
                                       progress: float, frame_idx: int) -> np.ndarray:
        """断面図用の詳細テキストを追加"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            info_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        # タイトル
        draw.text((10, 5), "Cross Section", fill=(0, 0, 0), font=title_font)
        
        # 断面情報
        axis_name = f"{section_axis.upper()}-Axis"
        draw.text((10, 25), f"Plane: {axis_name}", fill=(50, 50, 50), font=info_font)
        draw.text((10, 40), f"Position: {section_pos:.3f}", fill=(50, 50, 50), font=info_font)
        draw.text((10, 55), f"Range: [{section_min:.2f}, {section_max:.2f}]", fill=(100, 100, 100), font=info_font)
        
        # 進行状況バー
        bar_x, bar_y = 10, 75
        bar_width, bar_height = 100, 8
        
        # 背景バー
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                      fill=(200, 200, 200), outline=(150, 150, 150))
        
        # 進行バー
        progress_width = int(bar_width * progress / 100)
        draw.rectangle([bar_x, bar_y, bar_x + progress_width, bar_y + bar_height], 
                      fill=(50, 150, 200))
        
        # 進行率テキスト
        draw.text((bar_x + bar_width + 5, bar_y - 2), f"{progress:.0f}%", 
                 fill=(50, 50, 50), font=info_font)
        
        # フレーム番号
        draw.text((img.shape[1] - 50, img.shape[0] - 20), f"Frame {frame_idx + 1}", 
                 fill=(100, 100, 100), font=info_font, anchor="rb")
        
        return np.array(pil_img)
    
    def _add_no_section_message(self, img: np.ndarray) -> np.ndarray:
        """断面がない場合のメッセージを追加"""
        from PIL import Image, ImageDraw, ImageFont
        
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # 中央にメッセージを表示
        text = "No points in this section"
        text_width = draw.textlength(text, font=font)
        x = (img.shape[1] - text_width) // 2
        y = img.shape[0] // 2
        
        draw.text((x, y), text, fill=(150, 150, 150), font=font)
        
        return np.array(pil_img)
    
    def _project_point_perspective(self, vertex: np.ndarray, camera_pos: np.ndarray,
                                 view_vector: np.ndarray, right_vector: np.ndarray,
                                 up_vector: np.ndarray, fov: float, aspect_ratio: float,
                                 image_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """点を透視投影"""
        relative_point = vertex - camera_pos
        z_distance = np.dot(relative_point, view_vector)
        
        if z_distance <= 0.1:
            return None
        
        x_cam = np.dot(relative_point, right_vector)
        y_cam = np.dot(relative_point, up_vector)
        
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        x_proj = f * x_cam / z_distance / aspect_ratio
        y_proj = f * y_cam / z_distance
        
        pixel_x = int((x_proj + 1.0) * 0.5 * image_size[0])
        pixel_y = int((1.0 - y_proj) * 0.5 * image_size[1])
        
        if 0 <= pixel_x < image_size[0] and 0 <= pixel_y < image_size[1]:
            return (pixel_x, pixel_y)
        
        return None
    
    def _draw_line(self, img: np.ndarray, start: Tuple[int, int], 
                  end: Tuple[int, int], color: List[int]):
        """画像に線を描画"""
        cv2.line(img, start, end, color, 1)
    
    def create_video_from_frames(self, frames: List[np.ndarray], 
                               output_path: str, fps: int = 30):
        """フレームからMP4動画を作成"""
        if not frames:
            print("No frames to create video")
            return
        
        height, width = frames[0].shape[:2]
        
        # OpenCVのVideoWriterを使用
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # RGB to BGR conversion for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved: {output_path}")
    
    def create_combined_frames(self, mesh: trimesh.Trimesh, points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
                              num_frames: int = 60, image_size: Tuple[int, int] = (800, 600),
                              rotation_axes: List[str] = ['x', 'y', 'z'], include_cross_sections: bool = True) -> List[np.ndarray]:
        """統合動画のフレームを作成（グリッドレイアウト）"""
        frames = []
        
        # グリッドサイズを計算
        grid_cols = 3 if include_cross_sections else 2  # ワイヤーフレーム、点群、(断面図)
        grid_rows = len(rotation_axes)  # 回転軸の数
        
        cell_width = image_size[0] // grid_cols
        cell_height = image_size[1] // grid_rows
        combined_width = image_size[0]
        combined_height = image_size[1]
        
        # 各軸の各タイプのフレームを事前に生成
        all_frames = {}
        
        for axis in rotation_axes:
            print(f"Generating frames for {axis}-axis...")
            
            # ワイヤーフレームフレーム
            wireframe_frames = self.create_wireframe_rotation_frames(
                mesh, num_frames, (cell_width, cell_height), rotation_axis=axis
            )
            
            # 点群フレーム
            pointcloud_frames = self.create_pointcloud_rotation_frames(
                points, normals, colors, num_frames, (cell_width, cell_height), rotation_axis=axis
            )
            
            # 断面図フレーム（メッシュも渡す）
            cross_section_frames = []
            if include_cross_sections:
                cross_section_frames = self.create_cross_section_frames(
                    mesh, points, normals, colors, num_frames, (cell_width, cell_height), section_axis=axis
                )
            
            all_frames[axis] = {
                'wireframe': wireframe_frames,
                'pointcloud': pointcloud_frames,
                'cross_section': cross_section_frames
            }
        
        # 統合フレームを作成
        print("Combining frames into grid layout...")
        for frame_idx in range(num_frames):
            # 統合画像を作成
            combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
            
            # グリッドに配置
            for row, axis in enumerate(rotation_axes):
                y_start = row * cell_height
                y_end = (row + 1) * cell_height
                
                # ワイヤーフレーム (左列)
                x_start = 0
                x_end = cell_width
                wireframe_img = cv2.resize(all_frames[axis]['wireframe'][frame_idx], 
                                        (cell_width, cell_height))
                combined_img[y_start:y_end, x_start:x_end] = wireframe_img
                
                # 点群 (中央列)
                x_start = cell_width
                x_end = 2 * cell_width
                pointcloud_img = cv2.resize(all_frames[axis]['pointcloud'][frame_idx], 
                                          (cell_width, cell_height))
                combined_img[y_start:y_end, x_start:x_end] = pointcloud_img
                
                # 断面図 (右列) - オプション
                if include_cross_sections:
                    x_start = 2 * cell_width
                    x_end = 3 * cell_width
                    cross_section_img = cv2.resize(all_frames[axis]['cross_section'][frame_idx], 
                                                 (cell_width, cell_height))
                    combined_img[y_start:y_end, x_start:x_end] = cross_section_img
            
            # グリッド線とラベルを追加
            combined_img = self._add_grid_labels(combined_img, rotation_axes, 
                                                cell_width, cell_height, frame_idx)
            
            frames.append(combined_img)
        
        return frames
    
    # ズーム機能は削除済み
    def _add_grid_labels(self, img: np.ndarray, rotation_axes: List[str],
                        cell_width: int, cell_height: int, frame_idx: int) -> np.ndarray:
        """グリッドにラベルと線を追加"""
        from PIL import Image, ImageDraw, ImageFont
        
        # NumPy配列をPIL Imageに変換
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # グリッド線を描画
        grid_color = (100, 100, 100)
        
        # 縦線
        for col in range(1, 3):
            x = col * cell_width
            draw.line([(x, 0), (x, img.shape[0])], fill=grid_color, width=2)
        
        # 横線
        for row in range(1, len(rotation_axes)):
            y = row * cell_height
            draw.line([(0, y), (img.shape[1], y)], fill=grid_color, width=2)
        
        # ヘッダーラベル
        header_labels = ["Wireframe", "Point Cloud", "Cross Section"]
        for col, label in enumerate(header_labels):
            x = col * cell_width + cell_width // 2
            draw.text((x, 5), label, fill=(0, 0, 0), font=font, anchor="ma")
        
        # 軸ラベル
        for row, axis in enumerate(rotation_axes):
            y = row * cell_height + cell_height // 2
            draw.text((5, y), f"{axis.upper()}-Axis", fill=(0, 0, 0), font=font, anchor="lm")
        
        # フレーム番号
        draw.text((img.shape[1] - 5, img.shape[0] - 5), f"Frame {frame_idx + 1}", 
                 fill=(0, 0, 0), font=small_font, anchor="rb")
        
        # PIL ImageをNumPy配列に戻す
        return np.array(pil_img)
    
    def create_rotation_videos(self, model_path: str, 
                             num_frames: int = 60, fps: int = 30,
                             rotation_axes: List[str] = ['x', 'y', 'z'],
                             include_cross_sections: bool = True,
                             create_combined: bool = True,
                             create_individual: bool = False,
                             video_resolution: Tuple[int, int] = (1920, 1080)):
        """回転動画と断面図動画を作成"""
        # モデルを読み込み
        mesh = self.load_model(model_path)
        if mesh is None:
            return
        
        model_name = Path(model_path).stem
        
        # 点群を抽出（点数を調整）
        point_count = min(3000, len(mesh.vertices) // 3)  # 大きなモデルでは点数を制限
        points, normals, colors = self.extract_pointcloud(mesh, point_count)
        
        print(f"Creating videos for {model_name}")
        print(f"Mesh vertices: {len(mesh.vertices)}")
        print(f"Point cloud points: {len(points)}")
        
        # X,Y,Z軸統合動画を作成（常に全軸を含む）
        if create_combined:
            print("Creating comprehensive X,Y,Z axis analysis video...")
            combined_frames = self.create_combined_frames(
                mesh, points, normals, colors, num_frames, 
                image_size=video_resolution, rotation_axes=['x', 'y', 'z'],  # 常に全軸
                include_cross_sections=include_cross_sections
            )
            combined_video_path = str(self.output_dir / f"{model_name}_xyz_analysis.mp4")
            self.create_video_from_frames(combined_frames, combined_video_path, fps=60)  # 高品質FPS
            
            # ズーム詳細動画は削除済み
        
        # 個別動画を作成（オプション）
        if create_individual:
            for axis in rotation_axes:
                print(f"Creating {axis}-axis individual videos...")
                
                # ワイヤーフレーム回転動画
                wireframe_frames = self.create_wireframe_rotation_frames(
                    mesh, num_frames, rotation_axis=axis
                )
                wireframe_video_path = str(self.output_dir / f"{model_name}_wireframe_rotation_{axis}.mp4")
                self.create_video_from_frames(wireframe_frames, wireframe_video_path, fps)
                
                # 点群回転動画
                pointcloud_frames = self.create_pointcloud_rotation_frames(
                    points, normals, colors, num_frames, rotation_axis=axis
                )
                pointcloud_video_path = str(self.output_dir / f"{model_name}_pointcloud_rotation_{axis}.mp4")
                self.create_video_from_frames(pointcloud_frames, pointcloud_video_path, fps)
            
            # 断面図動画を作成
            if include_cross_sections:
                for axis in rotation_axes:
                    print(f"Creating {axis}-axis cross-section videos...")
                    
                    cross_section_frames = self.create_cross_section_frames(
                        mesh, points, normals, colors, num_frames, section_axis=axis
                    )
                    cross_section_video_path = str(self.output_dir / f"{model_name}_cross_section_{axis}.mp4")
                    self.create_video_from_frames(cross_section_frames, cross_section_video_path, fps)
        
        if create_combined:
            print(f"\n✓ Comprehensive analysis video created: {model_name}_xyz_analysis.mp4")
        if create_individual:
            print(f"\n✓ Individual videos also created")
        print(f"\nAll videos saved in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="3DモデルのGemini分析用動画作成 - 高品質多角度解析")
    parser.add_argument("model_path", help="3Dモデルファイルのパス")
    parser.add_argument("--frames", type=int, default=600, help="フレーム数 (デフォルト: 600)")
    parser.add_argument("--fps", type=int, default=60, help="FPS (デフォルト: 60)")
    parser.add_argument("--axes", nargs='+', default=['x', 'y', 'z'], 
                       choices=['x', 'y', 'z'], help="個別動画用回転軸 (統合動画は常にX,Y,Z全軸)")
    parser.add_argument("--output", help="出力ディレクトリ")
    parser.add_argument("--scale-reference", type=float, help="スケールリファレンスサイズ (単位)")
    parser.add_argument("--resolution", choices=['720p', '1080p', '4k'], default='1080p',
                       help="動画解像度 (デフォルト: 1080p)")
    parser.add_argument("--no-cross-sections", action="store_true", 
                       help="断面図動画を作成しない")
    parser.add_argument("--individual", action="store_true", 
                       help="個別動画も作成する")
    parser.add_argument("--no-combined", action="store_true", 
                       help="統合動画を作成しない")
    # ズーム機能は削除済み
    
    args = parser.parse_args()
    
    # 解像度設定
    resolution_map = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '4k': (3840, 2160)
    }
    video_resolution = resolution_map[args.resolution]
    
    # 動画作成クラスを初期化
    video_creator = Model3DVideoCreator(args.output, args.scale_reference)
    
    # Gemini分析用高品質動画を作成
    video_creator.create_rotation_videos(
        args.model_path,
        num_frames=args.frames,
        fps=args.fps,
        rotation_axes=args.axes,  # 個別動画用
        include_cross_sections=not args.no_cross_sections,
        create_combined=not args.no_combined,
        create_individual=args.individual,
        # ズーム機能は削除済み
        video_resolution=video_resolution
    )

if __name__ == "__main__":
    main()
