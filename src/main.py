import os
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


class Config:
    def __init__(self,
                 image_dir_path: str,
                 target_image_name: str,
                 target_size: int,
                 num_points: int):
        
        self.target_image_path = os.path.join(image_dir_path, target_image_name)
        self.target_size = target_size
        self.num_points = num_points

        self.target_image = None
        self.target_image_vec_repr = None
        self.read_and_preprocess_image()

        self.basis_vectors_df = None
        self.get_basis_vectors()

    def read_and_preprocess_image(self):
        # 画像を読み込み
        img = Image.open(self.target_image_path)

        # 画像をグレースケールに変換
        img = img.convert('L')

        # 画像を正方形に切り抜く
        size = min(img.size)
        img = ImageOps.fit(img, (size, size), Image.LANCZOS)

        # 円形に切り抜くためのマスクを作成
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)

        # マスクを使って画像を円形に切り抜く
        img = Image.composite(img, Image.new('L', img.size, 255), mask)

        # 指定されたサイズにリサイズ
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)

        self.target_image = img
        self.target_image_vec_repr = np.array(img).flatten()
        
    def _approximate_circle_points(self, target_size, num_points):
        # 正方形に内接する円の中心座標と半径を計算
        center = (target_size // 2, target_size // 2)
        radius = target_size // 2

        points = []
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            x = center[0] + int(radius * math.cos(angle))
            y = center[1] + int(radius * math.sin(angle))
            points.append((x, y))

        return points

    def _get_all_combinations(self, points):
        combinations = []

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                combinations.append((points[i], points[j]))

        return combinations

    def get_basis_vectors(self):
        circle_points = self._approximate_circle_points(self.target_size, self.num_points)
        coordinates_combinations = self._get_all_combinations(circle_points)

        vector_representations = []

        for combination in coordinates_combinations:
            img = Image.new('L', (self.target_size, self.target_size), 255)
            draw = ImageDraw.Draw(img)
            draw.line(combination, fill=0, width=1)

            vector_representations.append(np.array(img).flatten())

        basis_vectors_dict = {
            "coords": coordinates_combinations,
            "vec_repr": vector_representations
        }

        self.basis_vectors_df = pd.DataFrame(basis_vectors_dict)



# if __name__ == "__main__":

#     image_dir_path = './../images'
#     target_image_name = 'eye.jpg'
#     target_size = 300
#     num_points = 100

#     config = Config(image_dir_path, target_image_name, target_size, num_points)

#     config.target_image.show()