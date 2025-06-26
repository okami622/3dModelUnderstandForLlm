"""
Color generation utilities for 3D models
"""
import numpy as np
import colorsys
from typing import List, Tuple


class ColorGenerator:
    """3Dモデルの色生成を担当するクラス"""
    
    def generate_position_based_color(self, position: np.ndarray, bounds: np.ndarray) -> List[int]:
        """位置ベースの色生成"""
        try:
            # 正規化された位置を計算
            normalized_pos = self._normalize_position(position, bounds)
            
            # HSVカラースペースで色を生成
            hue = (normalized_pos[0] + normalized_pos[1] + normalized_pos[2]) / 3.0
            saturation = 0.8
            value = 0.9
            
            # HSVからRGBに変換
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # 0-255の範囲に変換
            return [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
            
        except Exception as e:
            print(f"Color generation error: {e}")
            return [128, 128, 128]  # グレーをデフォルトに
    
    def _normalize_position(self, position: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """位置座標を正規化"""
        min_coords = bounds[0]
        max_coords = bounds[1]
        
        # 範囲の幅
        ranges = max_coords - min_coords
        
        # ゼロ除算を避ける
        ranges = np.where(ranges == 0, 1, ranges)
        
        # 正規化
        normalized = (position - min_coords) / ranges
        
        # 0-1の範囲にクランプ
        return np.clip(normalized, 0, 1)
    
    def apply_depth_shading(self, color: List[int], depth: float, max_depth: float) -> List[int]:
        """深度に基づくシェーディング"""
        try:
            if max_depth <= 0:
                return color
            
            # 深度の正規化（0-1の範囲）
            normalized_depth = min(depth / max_depth, 1.0)
            
            # 深度に基づく輝度調整（遠いほど暗く）
            brightness_factor = 1.0 - (normalized_depth * 0.3)
            
            # 色に適用
            shaded_color = [
                max(0, min(255, int(color[0] * brightness_factor))),
                max(0, min(255, int(color[1] * brightness_factor))),
                max(0, min(255, int(color[2] * brightness_factor)))
            ]
            
            return shaded_color
            
        except Exception as e:
            print(f"Depth shading error: {e}")
            return color
    
    def generate_gradient_color(self, progress: float, color_scheme: str = "rainbow") -> List[int]:
        """進行度に基づくグラデーション色生成"""
        try:
            # 進行度を0-1の範囲にクランプ
            progress = max(0.0, min(1.0, progress))
            
            if color_scheme == "rainbow":
                # レインボーカラー
                hue = progress * 0.8  # 赤から紫まで
                saturation = 0.9
                value = 0.9
                
            elif color_scheme == "heat":
                # ヒートマップカラー（青→緑→黄→赤）
                if progress < 0.33:
                    hue = 0.67 - (progress * 0.67 / 0.33)  # 青→緑
                elif progress < 0.67:
                    hue = 0.33 - ((progress - 0.33) * 0.33 / 0.34)  # 緑→黄
                else:
                    hue = 0.0  # 黄→赤
                saturation = 0.9
                value = 0.9
                
            elif color_scheme == "cool":
                # クールカラー（青→シアン→緑）
                hue = 0.5 + (progress * 0.17)
                saturation = 0.8
                value = 0.9
                
            else:  # デフォルト: グレースケール
                gray_value = int(progress * 255)
                return [gray_value, gray_value, gray_value]
            
            # HSVからRGBに変換
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # 0-255の範囲に変換
            return [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
            
        except Exception as e:
            print(f"Gradient color generation error: {e}")
            return [128, 128, 128]
    
    def blend_colors(self, color1: List[int], color2: List[int], ratio: float) -> List[int]:
        """2つの色をブレンド"""
        try:
            ratio = max(0.0, min(1.0, ratio))
            
            blended = [
                int(color1[0] * (1 - ratio) + color2[0] * ratio),
                int(color1[1] * (1 - ratio) + color2[1] * ratio),
                int(color1[2] * (1 - ratio) + color2[2] * ratio)
            ]
            
            return blended
            
        except Exception as e:
            print(f"Color blending error: {e}")
            return color1