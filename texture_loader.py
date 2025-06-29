"""
3D model texture loading utilities
"""
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image


class TextureLoader:
    """3Dモデルのテクスチャ読み込みを担当するクラス"""
    
    def load_model_with_textures(self, model_path: str) -> Optional[trimesh.Trimesh]:
        """フォーマットに応じたテクスチャ付きモデル読み込み"""
        model_path = Path(model_path)
        
        try:
            # フォーマット別の処理
            if model_path.suffix.lower() == '.obj':
                return self._load_obj_with_textures(str(model_path))
            elif model_path.suffix.lower() in ['.gltf', '.glb']:
                return self._load_gltf_with_textures(str(model_path))
            elif model_path.suffix.lower() == '.dae':
                return self._load_dae_with_textures(str(model_path))
            elif model_path.suffix.lower() == '.ply':
                return self._load_ply_with_textures(str(model_path))
            elif model_path.suffix.lower() == '.3mf':
                return self._load_3mf_with_textures(str(model_path))
            else:
                # 通常読み込み + 汎用テクスチャ探索
                mesh = trimesh.load(str(model_path))
                return self._try_find_generic_textures(mesh, str(model_path))
                
        except Exception as e:
            print(f"Texture loading failed: {e}")
            return trimesh.load(str(model_path))
    
    def _load_obj_with_textures(self, obj_path: str) -> Optional[trimesh.Trimesh]:
        """OBJ + MTL テクスチャ読み込み"""
        print(f"Loading OBJ with textures: {obj_path}")
        
        try:
            mesh = trimesh.load(obj_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # MTLファイルを探す
            mtl_path = self._find_mtl_file(obj_path)
            if mtl_path:
                print(f"Found MTL file: {mtl_path}")
                mesh = self._apply_mtl_textures(mesh, mtl_path)
            else:
                print("No MTL file found")
            
            return mesh
            
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            return trimesh.load(obj_path)
    
    def _find_mtl_file(self, obj_path: str) -> Optional[Path]:
        """MTLファイルを探す"""
        obj_path = Path(obj_path)
        
        # 同名MTLファイル
        mtl_path = obj_path.with_suffix('.mtl')
        if mtl_path.exists():
            return mtl_path
        
        # 同名ディレクトリ内
        dir_path = obj_path.parent / obj_path.stem
        if dir_path.exists():
            for mtl_file in dir_path.glob("*.mtl"):
                return mtl_file
        
        # 親ディレクトリ内
        for mtl_file in obj_path.parent.glob("*.mtl"):
            return mtl_file
        
        return None
    
    def _apply_mtl_textures(self, mesh: trimesh.Trimesh, mtl_path: Path) -> trimesh.Trimesh:
        """MTLファイルからテクスチャを適用"""
        try:
            materials = self._parse_mtl_file(mtl_path)
            
            if materials:
                print(f"Found {len(materials)} materials")
                material_name = list(materials.keys())[0]
                material = materials[material_name]
                
                # テクスチャマップを適用
                if 'map_Kd' in material:
                    texture_path = mtl_path.parent / material['map_Kd']
                    if texture_path.exists():
                        print(f"Loading texture: {texture_path}")
                        mesh = self._apply_texture_to_mesh(mesh, texture_path)
                
                # 拡散反射色を適用
                elif 'Kd' in material:
                    print(f"Applying diffuse color: {material['Kd']}")
                    mesh = self._apply_color_to_mesh(mesh, material['Kd'])
            
            return mesh
            
        except Exception as e:
            print(f"Error applying MTL: {e}")
            return mesh
    
    def _parse_mtl_file(self, mtl_path: Path) -> Dict:
        """MTLファイル解析"""
        materials = {}
        current_material = None
        
        try:
            with open(mtl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    command = parts[0]
                    
                    if command == 'newmtl':
                        current_material = parts[1]
                        materials[current_material] = {}
                    
                    elif current_material and command in ['Kd', 'Ka', 'Ks']:
                        if len(parts) >= 4:
                            materials[current_material][command] = [
                                float(parts[1]), float(parts[2]), float(parts[3])
                            ]
                    
                    elif current_material and command == 'map_Kd':
                        materials[current_material][command] = parts[1]
        
        except Exception as e:
            print(f"MTL parsing error: {e}")
        
        return materials
    
    def _load_gltf_with_textures(self, gltf_path: str) -> Optional[trimesh.Trimesh]:
        """GLTF/GLB テクスチャ読み込み"""
        print(f"Loading GLTF with textures: {gltf_path}")
        
        try:
            mesh = trimesh.load(gltf_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # 埋め込みテクスチャチェック
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                material = mesh.visual.material
                if hasattr(material, 'baseColorTexture') or hasattr(material, 'image'):
                    print("GLTF has embedded texture")
                    return mesh
            
            # 外部テクスチャファイルを探す
            mesh = self._try_find_generic_textures(mesh, gltf_path)
            return mesh
            
        except Exception as e:
            print(f"Error loading GLTF: {e}")
            return trimesh.load(gltf_path)
    
    def _load_dae_with_textures(self, dae_path: str) -> Optional[trimesh.Trimesh]:
        """DAE (Collada) テクスチャ読み込み"""
        print(f"Loading DAE with textures: {dae_path}")
        
        try:
            mesh = trimesh.load(dae_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # XML内のテクスチャ参照を解析
            dae_path = Path(dae_path)
            
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(dae_path)
                root = tree.getroot()
                
                # library_images からテクスチャを探す
                for image in root.findall('.//image'):
                    init_from = image.find('init_from')
                    if init_from is not None and init_from.text:
                        texture_path = dae_path.parent / init_from.text
                        if texture_path.exists():
                            print(f"Found DAE texture: {texture_path}")
                            return self._apply_texture_to_mesh(mesh, texture_path)
                        
                        # ファイル名のみの場合
                        texture_name = Path(init_from.text).name
                        texture_path = dae_path.parent / texture_name
                        if texture_path.exists():
                            print(f"Found DAE texture: {texture_path}")
                            return self._apply_texture_to_mesh(mesh, texture_path)
                            
            except Exception as xml_error:
                print(f"DAE XML parsing failed: {xml_error}")
            
            # フォールバック
            return self._try_find_generic_textures(mesh, str(dae_path))
            
        except Exception as e:
            print(f"Error loading DAE: {e}")
            return trimesh.load(dae_path)
    
    def _load_ply_with_textures(self, ply_path: str) -> Optional[trimesh.Trimesh]:
        """PLY テクスチャ読み込み"""
        print(f"Loading PLY with textures: {ply_path}")
        
        try:
            mesh = trimesh.load(ply_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            return self._try_find_generic_textures(mesh, ply_path)
            
        except Exception as e:
            print(f"Error loading PLY: {e}")
            return trimesh.load(ply_path)
    
    def _load_3mf_with_textures(self, mf3_path: str) -> Optional[trimesh.Trimesh]:
        """3MF テクスチャ読み込み"""
        print(f"Loading 3MF with textures: {mf3_path}")
        
        try:
            mesh = trimesh.load(mf3_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # ZIP内のテクスチャを探す
            try:
                import zipfile
                import tempfile
                
                with zipfile.ZipFile(mf3_path, 'r') as zip_file:
                    for file_name in zip_file.namelist():
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            print(f"Found 3MF texture: {file_name}")
                            
                            # 一時ファイルに抽出
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as temp_file:
                                temp_file.write(zip_file.read(file_name))
                                temp_path = Path(temp_file.name)
                            
                            try:
                                mesh = self._apply_texture_to_mesh(mesh, temp_path)
                            finally:
                                temp_path.unlink(missing_ok=True)
                            break
                            
            except Exception as zip_error:
                print(f"3MF texture extraction failed: {zip_error}")
            
            return self._try_find_generic_textures(mesh, mf3_path)
            
        except Exception as e:
            print(f"Error loading 3MF: {e}")
            return trimesh.load(mf3_path)
    
    def _try_find_generic_textures(self, mesh: trimesh.Trimesh, model_path: str) -> trimesh.Trimesh:
        """汎用テクスチャファイル探索"""
        if mesh is None:
            return mesh
            
        model_path = Path(model_path)
        model_name = model_path.stem
        
        # テクスチャファイル名パターン
        patterns = [
            f"{model_name}.jpg", f"{model_name}.png",
            f"{model_name}_diffuse.jpg", f"{model_name}_diffuse.png",
            f"{model_name}_albedo.jpg", f"{model_name}_color.png",
            f"{model_name}_texture.jpg", "texture.jpg", "diffuse.png"
        ]
        
        # 検索ディレクトリ
        search_dirs = [
            model_path.parent,
            model_path.parent / "textures",
            model_path.parent / "images",
            model_path.parent / model_name
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for pattern in patterns:
                texture_path = search_dir / pattern
                if texture_path.exists():
                    print(f"Found generic texture: {texture_path}")
                    try:
                        return self._apply_texture_to_mesh(mesh, texture_path)
                    except Exception as e:
                        print(f"Failed to apply texture: {e}")
                        continue
        
        print("No external textures found")
        return mesh
    
    def _apply_texture_to_mesh(self, mesh: trimesh.Trimesh, texture_path: Path) -> trimesh.Trimesh:
        """テクスチャ画像をメッシュに適用（強化版）"""
        try:
            texture_image = Image.open(texture_path).convert('RGB')
            texture_array = np.array(texture_image)
            
            # コントラストを強化
            texture_array = self._enhance_texture_contrast(texture_array)
            
            print(f"Texture loaded and enhanced: {texture_array.shape}")
            
            # UV座標がある場合
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                vertex_colors = self._map_texture_to_vertices(mesh.visual.uv, texture_array)
                mesh.visual.vertex_colors = vertex_colors
                print(f"Applied enhanced texture to {len(vertex_colors)} vertices")
            else:
                # 平均色を使用
                avg_color = np.mean(texture_array.reshape(-1, 3), axis=0)
                mesh = self._apply_color_to_mesh(mesh, avg_color / 255.0)
            
            return mesh
            
        except Exception as e:
            print(f"Error applying texture: {e}")
            return mesh
    
    def _map_texture_to_vertices(self, uv_coords: np.ndarray, texture_array: np.ndarray) -> np.ndarray:
        """UV座標を使ってテクスチャを頂点色にマッピング"""
        vertex_colors = []
        h, w = texture_array.shape[:2]
        
        for uv in uv_coords:
            u, v = uv[0], 1.0 - uv[1]  # Vを反転
            x = int(np.clip(u * (w - 1), 0, w - 1))
            y = int(np.clip(v * (h - 1), 0, h - 1))
            
            color = texture_array[y, x]
            vertex_colors.append([color[0], color[1], color[2], 255])
        
        return np.array(vertex_colors)
    
    def _apply_color_to_mesh(self, mesh: trimesh.Trimesh, color: List[float]) -> trimesh.Trimesh:
        """単色をメッシュに適用"""
        try:
            if np.max(color) <= 1.0:
                color_255 = [int(c * 255) for c in color]
            else:
                color_255 = [int(c) for c in color]
            
            vertex_colors = np.full((len(mesh.vertices), 4), 
                                  [color_255[0], color_255[1], color_255[2], 255])
            mesh.visual.vertex_colors = vertex_colors
            
            print(f"Applied color {color_255} to {len(mesh.vertices)} vertices")
            return mesh
            
        except Exception as e:
            print(f"Error applying color: {e}")
            return mesh
    
    def _enhance_texture_contrast(self, texture_array: np.ndarray, factor: float = 1.3) -> np.ndarray:
        """テクスチャのコントラストを強化"""
        try:
            # 0-1の範囲に正規化
            normalized = texture_array.astype(np.float32) / 255.0
            
            # コントラスト強化（S字カーブ）
            enhanced = np.power(normalized, 1.0 / factor)
            
            # 0-255の範囲に戻す
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing contrast: {e}")
            return texture_array