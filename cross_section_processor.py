"""
Cross-section processing utilities for 3D models
"""
import numpy as np
import trimesh
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class CrossSectionProcessor:
    """3Dモデルの断面処理を担当するクラス"""
    
    def __init__(self):
        self.background_color = [248, 248, 248]
        self.line_color = [100, 100, 100]
        self.fill_color = [200, 200, 255]
        self.line_width = 2.0
    
    def create_cross_sections(self, mesh: trimesh.Trimesh, axis: str = 'z', 
                            num_sections: int = 10) -> List[np.ndarray]:
        """メッシュの断面を作成"""
        print(f"Creating {num_sections} cross-sections along {axis} axis")
        
        sections = []
        bounds = mesh.bounds
        
        # 軸の選択
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        # 断面の位置を計算（5%マージンを設けて範囲を設定）
        min_val = bounds[0, axis_index]
        max_val = bounds[1, axis_index]
        margin = (max_val - min_val) * 0.05
        
        start_pos = min_val + margin
        end_pos = max_val - margin
        
        positions = np.linspace(start_pos, end_pos, num_sections)
        
        for i, pos in enumerate(positions):
            try:
                # 断面を計算
                section = self._create_single_section(mesh, axis, pos)
                
                if section is not None:
                    sections.append(section)
                    print(f"Section {i+1}/{num_sections}: {len(section)} contours at {axis}={pos:.3f}")
                else:
                    print(f"Section {i+1}/{num_sections}: No intersection at {axis}={pos:.3f}")
                    
            except Exception as e:
                print(f"Error creating section {i+1}: {e}")
                continue
        
        print(f"Successfully created {len(sections)} cross-sections")
        return sections
    
    def _create_single_section(self, mesh: trimesh.Trimesh, axis: str, 
                             position: float) -> Optional[np.ndarray]:
        """単一の断面を作成"""
        try:
            # 断面平面を定義
            plane_normal = np.zeros(3)
            axis_index = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
            plane_normal[axis_index] = 1.0
            
            plane_origin = np.zeros(3)
            plane_origin[axis_index] = position
            
            # メッシュと平面の交線を計算
            section_2d = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            
            if section_2d is None:
                return None
            
            # 2D断面を描画用の画像配列に変換
            section_image = self._render_section_to_array(section_2d, mesh.bounds, axis)
            
            return section_image
            
        except Exception as e:
            print(f"Error creating single section: {e}")
            return None
    
    def _render_section_to_array(self, section_2d, mesh_bounds: np.ndarray, 
                                axis: str, image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """2D断面を画像配列に描画"""
        try:
            # 断面の境界を計算
            if hasattr(section_2d, 'entities') and len(section_2d.entities) > 0:
                # Path2Dオブジェクトの場合
                vertices = section_2d.vertices
                
                if len(vertices) == 0:
                    return self._create_empty_section_image(image_size)
                
                # 軸に応じた2D座標の抽出
                if axis.lower() == 'x':
                    coords_2d = vertices[:, [1, 2]]  # Y-Z平面
                elif axis.lower() == 'y':
                    coords_2d = vertices[:, [0, 2]]  # X-Z平面
                else:  # z軸
                    coords_2d = vertices[:, [0, 1]]  # X-Y平面
                
                # 描画
                section_image = self._draw_section_contours(coords_2d, section_2d.entities, 
                                                          mesh_bounds, axis, image_size)
                
                return section_image
            else:
                return self._create_empty_section_image(image_size)
                
        except Exception as e:
            print(f"Error rendering section: {e}")
            return self._create_empty_section_image(image_size)
    
    def _draw_section_contours(self, coords_2d: np.ndarray, entities, mesh_bounds: np.ndarray,
                             axis: str, image_size: Tuple[int, int]) -> np.ndarray:
        """断面の輪郭を描画"""
        try:
            # Matplotlibで描画
            fig, ax = plt.subplots(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
            ax.set_aspect('equal')
            
            # 背景色を設定
            bg_color = [c/255.0 for c in self.background_color]
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            
            # 輪郭線の描画
            line_color = [c/255.0 for c in self.line_color]
            fill_color = [c/255.0 for c in self.fill_color]
            
            polygons = []
            
            for entity in entities:
                if hasattr(entity, 'points'):
                    # 線分の場合
                    points = coords_2d[entity.points]
                    ax.plot(points[:, 0], points[:, 1], 
                           color=line_color, linewidth=self.line_width)
                    
                    # 閉じた輪郭の場合はポリゴンとして追加
                    if len(points) > 2:
                        polygon = Polygon(points, closed=True, 
                                        facecolor=fill_color, alpha=0.3,
                                        edgecolor=line_color, linewidth=self.line_width)
                        polygons.append(polygon)
            
            # ポリゴンを追加
            if polygons:
                collection = PatchCollection(polygons, match_original=True)
                ax.add_collection(collection)
            
            # 軸の範囲を設定
            if axis.lower() == 'x':
                ax.set_xlim(mesh_bounds[0, 1], mesh_bounds[1, 1])
                ax.set_ylim(mesh_bounds[0, 2], mesh_bounds[1, 2])
                ax.set_xlabel('Y')
                ax.set_ylabel('Z')
            elif axis.lower() == 'y':
                ax.set_xlim(mesh_bounds[0, 0], mesh_bounds[1, 0])
                ax.set_ylim(mesh_bounds[0, 2], mesh_bounds[1, 2])
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
            else:  # z軸
                ax.set_xlim(mesh_bounds[0, 0], mesh_bounds[1, 0])
                ax.set_ylim(mesh_bounds[0, 1], mesh_bounds[1, 1])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            
            # 軸とタイトルを非表示
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('')
            
            # 余白を削除
            plt.tight_layout(pad=0)
            
            # 画像配列に変換
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            print(f"Error drawing section contours: {e}")
            return self._create_empty_section_image(image_size)
    
    def _create_empty_section_image(self, image_size: Tuple[int, int]) -> np.ndarray:
        """空の断面画像を作成"""
        empty_image = np.full((image_size[1], image_size[0], 3), 
                            self.background_color, dtype=np.uint8)
        return empty_image
    
    def analyze_section_properties(self, sections: List[np.ndarray]) -> dict:
        """断面の特性を分析"""
        properties = {
            'num_sections': len(sections),
            'sizes': [],
            'complexities': []
        }
        
        for i, section in enumerate(sections):
            # 断面のサイズ（非背景ピクセル数）
            bg_color = np.array(self.background_color)
            non_bg_mask = ~np.all(section == bg_color, axis=2)
            size = np.sum(non_bg_mask)
            properties['sizes'].append(size)
            
            # 複雑度（エッジ数の近似）
            try:
                gray = np.mean(section, axis=2)
                edges = np.sum(np.abs(np.diff(gray, axis=0))) + np.sum(np.abs(np.diff(gray, axis=1)))
                properties['complexities'].append(edges)
            except:
                properties['complexities'].append(0)
        
        return properties