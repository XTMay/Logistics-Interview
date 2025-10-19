"""
数据生成器 - 生成模拟的物流包裹图像和标注数据
用于演示目的，真实项目中应使用实际采集的数据
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2


class PackageDataGenerator:
    """物流包裹数据生成器"""

    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, 'sample_images')
        self.annotation_dir = os.path.join(output_dir, 'annotations')

        # 创建目录
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)

        # 包裹类型和颜色
        self.package_types = ['box', 'envelope', 'irregular']
        self.colors = [
            (139, 90, 60),   # 棕色（纸箱）
            (255, 255, 240), # 白色（信封）
            (200, 200, 200), # 灰色
            (160, 120, 80),  # 深棕色
        ]

    def generate_background(self, width=800, height=600):
        """生成仓库背景"""
        # 创建灰色背景模拟传送带或仓库地面
        background = np.ones((height, width, 3), dtype=np.uint8) * random.randint(100, 150)

        # 添加噪声模拟真实环境
        noise = np.random.normal(0, 10, (height, width, 3))
        background = np.clip(background + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(background)

    def generate_box(self, size_range=(100, 300)):
        """生成箱子形状的包裹"""
        width = random.randint(*size_range)
        height = random.randint(size_range[0], int(width * 1.2))

        # 创建箱子图像
        box = Image.new('RGB', (width, height), random.choice(self.colors))
        draw = ImageDraw.Draw(box)

        # 添加边缘线条模拟纸箱折痕
        line_color = tuple(max(0, c - 40) for c in box.getpixel((width//2, height//2)))
        for i in range(5, width-5, random.randint(30, 60)):
            draw.line([(i, 0), (i, height)], fill=line_color, width=2)

        # 添加胶带（随机）
        if random.random() > 0.5:
            tape_y = height // 2
            draw.rectangle([(10, tape_y-5), (width-10, tape_y+5)],
                         fill=(210, 180, 140), outline=(180, 150, 110))

        return box, 'box'

    def generate_envelope(self, size_range=(150, 250)):
        """生成信封形状的包裹"""
        width = random.randint(*size_range)
        height = int(width * 0.7)  # 信封通常是长方形

        # 创建信封
        envelope = Image.new('RGB', (width, height), (255, 255, 240))
        draw = ImageDraw.Draw(envelope)

        # 添加信封边框
        draw.rectangle([(5, 5), (width-5, height-5)],
                      outline=(200, 200, 200), width=3)

        # 添加地址框模拟
        draw.rectangle([(width//4, height//3), (3*width//4, 2*height//3)],
                      outline=(150, 150, 150), width=1)

        return envelope, 'envelope'

    def generate_irregular(self, size_range=(120, 280)):
        """生成不规则包裹"""
        size = random.randint(*size_range)

        # 创建椭圆或多边形形状
        irregular = Image.new('RGB', (size, size), (200, 200, 200))
        draw = ImageDraw.Draw(irregular)

        # 绘制椭圆
        padding = size // 5
        draw.ellipse([(padding, padding), (size-padding, size-padding)],
                    fill=random.choice(self.colors))

        return irregular, 'irregular'

    def add_damage(self, image, damage_prob=0.3):
        """添加破损效果"""
        if random.random() > damage_prob:
            return image, False

        # 转换为OpenCV格式
        img_array = np.array(image)

        # 添加撕裂效果（黑色不规则线条）
        for _ in range(random.randint(2, 5)):
            x1, y1 = random.randint(0, image.width), random.randint(0, image.height)
            x2, y2 = x1 + random.randint(-50, 50), y1 + random.randint(-50, 50)
            cv2.line(img_array, (x1, y1), (x2, y2), (50, 50, 50),
                    thickness=random.randint(3, 8))

        # 添加污渍
        for _ in range(random.randint(1, 3)):
            cx, cy = random.randint(0, image.width), random.randint(0, image.height)
            radius = random.randint(10, 30)
            cv2.circle(img_array, (cx, cy), radius, (80, 80, 80), -1)

        return Image.fromarray(img_array), True

    def apply_lighting(self, image):
        """应用光照变化"""
        # 随机亮度调整
        enhancer = random.uniform(0.6, 1.4)
        img_array = np.array(image).astype(np.float32)
        img_array = np.clip(img_array * enhancer, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def generate_sample(self, sample_id, num_packages=None):
        """生成一个样本图像（包含1-3个包裹）"""
        if num_packages is None:
            num_packages = random.randint(1, 3)

        # 生成背景
        background = self.generate_background()
        width, height = background.size

        annotations = {
            'image_id': sample_id,
            'width': width,
            'height': height,
            'objects': []
        }

        # 生成包裹
        for i in range(num_packages):
            # 随机选择包裹类型
            package_type = random.choice(['box', 'envelope', 'irregular'])

            if package_type == 'box':
                package, pkg_type = self.generate_box()
            elif package_type == 'envelope':
                package, pkg_type = self.generate_envelope()
            else:
                package, pkg_type = self.generate_irregular()

            # 添加破损
            package, is_damaged = self.add_damage(package)

            # 应用光照
            package = self.apply_lighting(package)

            # 随机旋转
            angle = random.randint(-15, 15)
            package = package.rotate(angle, expand=True, fillcolor=(128, 128, 128))

            # 随机位置放置
            pkg_width, pkg_height = package.size
            max_x = max(0, width - pkg_width - 50)
            max_y = max(0, height - pkg_height - 50)

            x = random.randint(20, max_x) if max_x > 20 else 20
            y = random.randint(20, max_y) if max_y > 20 else 20

            # 粘贴到背景
            background.paste(package, (x, y), package if package.mode == 'RGBA' else None)

            # 记录标注
            annotations['objects'].append({
                'id': i,
                'category': pkg_type,
                'bbox': [x, y, x + pkg_width, y + pkg_height],
                'is_damaged': is_damaged,
                'area': pkg_width * pkg_height
            })

        return background, annotations

    def generate_dataset(self, num_samples=50):
        """生成完整数据集"""
        print(f"开始生成 {num_samples} 个样本...")

        all_annotations = []

        for i in range(num_samples):
            # 生成样本
            image, annotation = self.generate_sample(f"sample_{i:04d}")

            # 保存图像
            image_path = os.path.join(self.image_dir, f"package_{i:04d}.jpg")
            image.save(image_path, quality=95)

            # 保存标注
            all_annotations.append(annotation)

            if (i + 1) % 10 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本")

        # 保存所有标注到JSON文件
        annotation_path = os.path.join(self.annotation_dir, 'annotations.json')
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, indent=2, ensure_ascii=False)

        print(f"\n数据生成完成！")
        print(f"图像保存在: {self.image_dir}")
        print(f"标注保存在: {annotation_path}")

        # 生成统计信息
        self.print_statistics(all_annotations)

        return all_annotations

    def print_statistics(self, annotations):
        """打印数据集统计信息"""
        total_objects = sum(len(ann['objects']) for ann in annotations)

        category_count = {}
        damage_count = 0

        for ann in annotations:
            for obj in ann['objects']:
                cat = obj['category']
                category_count[cat] = category_count.get(cat, 0) + 1
                if obj['is_damaged']:
                    damage_count += 1

        print("\n" + "="*50)
        print("数据集统计")
        print("="*50)
        print(f"总图像数: {len(annotations)}")
        print(f"总包裹数: {total_objects}")
        print(f"\n类别分布:")
        for cat, count in sorted(category_count.items()):
            print(f"  {cat}: {count} ({count/total_objects*100:.1f}%)")
        print(f"\n破损包裹: {damage_count} ({damage_count/total_objects*100:.1f}%)")
        print("="*50)


if __name__ == '__main__':
    # 生成数据集
    generator = PackageDataGenerator(output_dir='data')
    generator.generate_dataset(num_samples=50)
