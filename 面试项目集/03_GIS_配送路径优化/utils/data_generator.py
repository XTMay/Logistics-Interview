"""
数据生成器 - 生成配送订单和仓库位置数据
用于演示目的
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class DeliveryDataGenerator:
    """配送数据生成器"""

    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 城市中心（模拟北京）
        self.city_center = {'lat': 39.9042, 'lon': 116.4074}

        # 城市范围（约±0.3度，约30km范围）
        self.lat_range = 0.3
        self.lon_range = 0.3

        # 仓库位置
        self.warehouses = [
            {'id': 'WH001', 'name': '北京东城仓', 'lat': 39.93, 'lon': 116.42},
            {'id': 'WH002', 'name': '北京西城仓', 'lat': 39.91, 'lon': 116.38},
        ]

        # 区域名称（模拟北京区域）
        self.districts = ['朝阳区', '海淀区', '丰台区', '石景山区', '东城区', '西城区']

    def generate_random_location(self, cluster_center=None):
        """生成随机位置（可选聚类）"""
        if cluster_center:
            # 在聚类中心附近生成
            lat = cluster_center['lat'] + np.random.normal(0, 0.02)
            lon = cluster_center['lon'] + np.random.normal(0, 0.02)
        else:
            # 完全随机
            lat = self.city_center['lat'] + random.uniform(-self.lat_range, self.lat_range)
            lon = self.city_center['lon'] + random.uniform(-self.lon_range, self.lon_range)

        return {'lat': lat, 'lon': lon}

    def generate_time_window(self, base_hour=9):
        """生成配送时间窗"""
        # 随机选择时间窗类型
        window_type = random.choice(['morning', 'afternoon', 'flexible'])

        if window_type == 'morning':
            start_hour = random.randint(9, 11)
            end_hour = start_hour + random.randint(2, 3)
        elif window_type == 'afternoon':
            start_hour = random.randint(13, 15)
            end_hour = start_hour + random.randint(2, 3)
        else:  # flexible
            start_hour = 9
            end_hour = 18

        start_time = f"{start_hour:02d}:00"
        end_time = f"{end_hour:02d}:00"

        return start_time, end_time

    def generate_orders(self, num_orders=50, clustering=True):
        """生成订单数据"""
        print(f"生成 {num_orders} 个订单...")

        orders = []

        # 如果使用聚类，生成几个聚类中心
        if clustering:
            num_clusters = random.randint(3, 5)
            cluster_centers = [
                self.generate_random_location() for _ in range(num_clusters)
            ]
        else:
            cluster_centers = None

        for i in range(num_orders):
            # 选择聚类中心（70%概率聚类，30%随机）
            if cluster_centers and random.random() < 0.7:
                center = random.choice(cluster_centers)
            else:
                center = None

            location = self.generate_random_location(center)

            # 生成订单信息
            start_time, end_time = self.generate_time_window()

            order = {
                'order_id': f'ORD{i+1:05d}',
                'latitude': round(location['lat'], 6),
                'longitude': round(location['lon'], 6),
                'district': random.choice(self.districts),
                'address': f'{random.choice(self.districts)}{random.choice(["建国路", "中关村", "王府井", "三里屯"])}{random.randint(1,999)}号',
                'weight_kg': round(random.uniform(0.5, 20), 1),
                'volume_cbm': round(random.uniform(0.01, 0.5), 2),
                'time_window_start': start_time,
                'time_window_end': end_time,
                'service_time_min': random.randint(5, 15),  # 服务时间（卸货等）
                'priority': random.choice(['normal', 'normal', 'normal', 'high']),  # 大部分普通
                'customer_name': f'客户{i+1:03d}',
                'phone': f'138{random.randint(10000000, 99999999)}',
            }

            orders.append(order)

        # 转换为DataFrame并保存
        df = pd.DataFrame(orders)
        output_path = os.path.join(self.output_dir, 'orders.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"✓ 订单数据已保存至 {output_path}")
        return df

    def save_warehouses(self):
        """保存仓库数据"""
        output_path = os.path.join(self.output_dir, 'warehouses.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.warehouses, f, indent=2, ensure_ascii=False)

        print(f"✓ 仓库数据已保存至 {output_path}")
        return self.warehouses

    def print_statistics(self, df):
        """打印数据统计"""
        print("\n" + "="*70)
        print("数据集统计")
        print("="*70)

        print(f"\n订单总数: {len(df)}")

        print(f"\n区域分布:")
        district_counts = df['district'].value_counts()
        for district, count in district_counts.items():
            print(f"  {district}: {count} ({count/len(df)*100:.1f}%)")

        print(f"\n重量统计:")
        print(f"  平均重量: {df['weight_kg'].mean():.2f} kg")
        print(f"  最大重量: {df['weight_kg'].max():.2f} kg")
        print(f"  最小重量: {df['weight_kg'].min():.2f} kg")

        print(f"\n时间窗分布:")
        morning = len(df[df['time_window_end'] <= '12:00'])
        afternoon = len(df[df['time_window_start'] >= '13:00'])
        flexible = len(df) - morning - afternoon
        print(f"  上午配送: {morning} ({morning/len(df)*100:.1f}%)")
        print(f"  下午配送: {afternoon} ({afternoon/len(df)*100:.1f}%)")
        print(f"  灵活时间: {flexible} ({flexible/len(df)*100:.1f}%)")

        print(f"\n优先级分布:")
        priority_counts = df['priority'].value_counts()
        for priority, count in priority_counts.items():
            print(f"  {priority}: {count} ({count/len(df)*100:.1f}%)")

        print(f"\n坐标范围:")
        print(f"  纬度: {df['latitude'].min():.4f} ~ {df['latitude'].max():.4f}")
        print(f"  经度: {df['longitude'].min():.4f} ~ {df['longitude'].max():.4f}")

        print("="*70 + "\n")

    def generate_all(self, num_orders=50):
        """生成所有数据"""
        print("\n" + "="*70)
        print("配送数据生成")
        print("="*70 + "\n")

        # 生成订单
        df = self.generate_orders(num_orders=num_orders)

        # 保存仓库
        warehouses = self.save_warehouses()

        # 打印统计
        self.print_statistics(df)

        return df, warehouses


if __name__ == '__main__':
    generator = DeliveryDataGenerator()
    generator.generate_all(num_orders=50)
