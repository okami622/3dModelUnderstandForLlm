"""
X3D file format loading utilities
"""
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, List
import xml.etree.ElementTree as ET


class X3DLoader:
    """X3Dファイル読み込みを担当するクラス"""
    
    def load_x3d_file(self, file_path: str) -> Optional[trimesh.Trimesh]:
        """X3Dファイルを読み込む"""
        print(f"Attempting to load X3D file: {file_path}")
        
        try:
            # X3Dファイルをパース
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            vertices = []
            faces = []
            
            # IndexedFaceSetを探す
            for indexed_face_set in root.iter():
                if 'IndexedFaceSet' in indexed_face_set.tag:
                    self._process_indexed_face_set(indexed_face_set, vertices, faces)
            
            # Shape > Geometry > IndexedFaceSetのパターンも確認
            for shape in root.iter():
                if 'Shape' in shape.tag:
                    for geometry in shape:
                        if 'Geometry' in geometry.tag or 'IndexedFaceSet' in geometry.tag:
                            self._extract_geometry(geometry, vertices, faces)
            
            if vertices and faces:
                vertices = np.array(vertices)
                faces = np.array(faces)
                
                # 無効な面インデックスを除去
                valid_faces = self._filter_valid_faces(faces, len(vertices))
                
                if valid_faces:
                    faces = np.array(valid_faces)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    print(f"✓ X3D converted to mesh: {len(vertices)} vertices, {len(faces)} faces")
                    return mesh
                else:
                    print("⚠️  No valid faces found in X3D file")
            else:
                print("⚠️  No geometry found in X3D file")
                
        except Exception as e:
            print(f"⚠️  Failed to load X3D file: {e}")
            
        return None
    
    def _process_indexed_face_set(self, indexed_face_set, vertices: List, faces: List):
        """IndexedFaceSetを処理"""
        # Coordinateノードを探す
        coordinate_node = None
        for child in indexed_face_set:
            if 'Coordinate' in child.tag:
                coordinate_node = child
                break
        
        if coordinate_node is not None and 'point' in coordinate_node.attrib:
            # 頂点座標を抽出
            points_str = coordinate_node.attrib['point']
            point_values = [float(x) for x in points_str.replace(',', ' ').split()]
            
            # 3つずつグループ化してx,y,z座標にする
            for i in range(0, len(point_values), 3):
                if i + 2 < len(point_values):
                    vertices.append([point_values[i], point_values[i+1], point_values[i+2]])
        
        # 面のインデックスを抽出
        if 'coordIndex' in indexed_face_set.attrib:
            indices_str = indexed_face_set.attrib['coordIndex']
            indices = [int(x) for x in indices_str.replace(',', ' ').split() if x.strip() and x.strip() != '-1']
            
            # 3つずつグループ化して三角形の面にする
            for i in range(0, len(indices), 3):
                if i + 2 < len(indices):
                    faces.append([indices[i], indices[i+1], indices[i+2]])
    
    def _extract_geometry(self, geometry_node, vertices: List, faces: List):
        """X3Dジオメトリノードから頂点と面を抽出"""
        try:
            # 直接的なIndexedFaceSet
            if 'IndexedFaceSet' in geometry_node.tag:
                self._process_indexed_face_set_advanced(geometry_node, vertices, faces)
            
            # 子ノードをチェック
            for child in geometry_node:
                if 'IndexedFaceSet' in child.tag:
                    self._process_indexed_face_set_advanced(child, vertices, faces)
                    
        except Exception as e:
            print(f"Warning: Error processing X3D geometry node: {e}")
    
    def _process_indexed_face_set_advanced(self, face_set_node, vertices: List, faces: List):
        """IndexedFaceSetノードの高度な処理"""
        try:
            coordinate_node = None
            
            # Coordinateノードを探す
            for child in face_set_node:
                if 'Coordinate' in child.tag:
                    coordinate_node = child
                    break
            
            # 頂点座標を抽出
            if coordinate_node is not None and 'point' in coordinate_node.attrib:
                points_str = coordinate_node.attrib['point']
                point_values = [float(x) for x in points_str.replace(',', ' ').split()]
                
                vertex_start_idx = len(vertices)
                
                # 3つずつグループ化
                for i in range(0, len(point_values), 3):
                    if i + 2 < len(point_values):
                        vertices.append([point_values[i], point_values[i+1], point_values[i+2]])
                
                # 面のインデックスを抽出
                if 'coordIndex' in face_set_node.attrib:
                    indices_str = face_set_node.attrib['coordIndex']
                    # -1で区切られた面のインデックス
                    face_groups = indices_str.replace(',', ' ').split('-1')
                    
                    for face_group in face_groups:
                        indices = [int(x) for x in face_group.split() if x.strip()]
                        
                        if len(indices) >= 3:
                            # 三角形に分割（ファン三角分割）
                            for i in range(1, len(indices) - 1):
                                face = [
                                    vertex_start_idx + indices[0],
                                    vertex_start_idx + indices[i],
                                    vertex_start_idx + indices[i + 1]
                                ]
                                faces.append(face)
                                
        except Exception as e:
            print(f"Warning: Error processing IndexedFaceSet: {e}")
    
    def _filter_valid_faces(self, faces: np.ndarray, num_vertices: int) -> List:
        """無効な面インデックスを除去"""
        valid_faces = []
        for face in faces:
            if all(0 <= idx < num_vertices for idx in face):
                valid_faces.append(face)
        return valid_faces