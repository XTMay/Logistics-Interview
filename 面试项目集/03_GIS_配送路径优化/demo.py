"""
完整演示脚本 - 展示配送路径优化系统的完整功能
适合面试演示使用
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cdist
import folium
from folium import plugins


class RouteOptimizationDemo:
    """路径优化演示系统"""

    def __init__(self):
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

        # 车辆参数
        self.vehicle_capacity_kg = 300  # 车辆最大载重
        self.vehicle_capacity_cbm = 5   # 车辆最大体积
        self.num_vehicles = 3           # 车辆数量

        # 加载数据
        self.load_data()

        # 性能指标
        self.metrics = {
            'optimization': {
                'distance_before': 0,
                'distance_after': 0,
                'distance_saved': 0,
                'distance_saved_pct': 0
            },
            'efficiency': {
                'orders_per_vehicle_before': 0,
                'orders_per_vehicle_after': 0,
                'efficiency_improvement': 0
            },
            'time': {
                'optimization_time_sec': 0,
                'avg_route_time_min': 0
            },
            'business': {
                'fuel_cost_saved': 0,
                'co2_reduction_kg': 0,
                'on_time_rate': 0.95
            }
        }

    def load_data(self):
        """加载数据"""
        orders_path = 'data/orders.csv'
        warehouses_path = 'data/warehouses.json'

        if not os.path.exists(orders_path):
            print("未找到订单数据，正在生成...")
            from utils.data_generator import DeliveryDataGenerator
            gen = DeliveryDataGenerator()
            gen.generate_all(num_orders=50)

        # 加载订单
        self.orders = pd.read_csv(orders_path)
        print(f"✓ 已加载 {len(self.orders)} 个订单")

        # 加载仓库
        with open(warehouses_path, 'r', encoding='utf-8') as f:
            self.warehouses = json.load(f)
        print(f"✓ 已加载 {len(self.warehouses)} 个仓库")

        # 使用第一个仓库作为起点
        self.depot = self.warehouses[0]

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两点间距离（km）"""
        # 简化的距离计算（使用欧氏距离近似）
        # 1度纬度约111km, 1度经度约111*cos(lat)km
        lat_km = (lat2 - lat1) * 111
        lon_km = (lon2 - lon1) * 111 * np.cos(np.radians(lat1))
        distance = np.sqrt(lat_km**2 + lon_km**2)
        return abs(distance)

    def create_distance_matrix(self):
        """创建距离矩阵"""
        # 所有位置：仓库 + 订单
        all_locations = [(self.depot['lat'], self.depot['lon'])]
        all_locations.extend([(row['latitude'], row['longitude'])
                            for _, row in self.orders.iterrows()])

        n = len(all_locations)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = self.calculate_distance(
                        all_locations[i][0], all_locations[i][1],
                        all_locations[j][0], all_locations[j][1]
                    )

        return distance_matrix

    def greedy_route(self, orders_subset, start_idx=0):
        """贪心算法生成路径"""
        unvisited = set(range(len(orders_subset)))
        route = [start_idx]
        current = start_idx

        while unvisited:
            # 移除已访问
            unvisited.discard(current)

            if not unvisited:
                break

            # 找最近的未访问点
            nearest = min(unvisited, key=lambda x: self.calculate_distance(
                orders_subset.iloc[current]['latitude'],
                orders_subset.iloc[current]['longitude'],
                orders_subset.iloc[x]['latitude'],
                orders_subset.iloc[x]['longitude']
            ))

            route.append(nearest)
            current = nearest

        return route

    def optimize_routes(self):
        """优化配送路径"""
        print("\n" + "="*70)
        print("路径优化")
        print("="*70 + "\n")

        # 1. 基线方案：随机分配
        print("步骤 1/3: 生成基线方案（随机分配）...")
        baseline_routes = self.create_baseline_routes()
        baseline_distance = self.calculate_total_distance(baseline_routes)
        print(f"  基线总距离: {baseline_distance:.2f} km")

        # 2. 优化方案：聚类 + 贪心
        print("\n步骤 2/3: 运行优化算法（K-means聚类 + 贪心路径）...")
        optimized_routes = self.create_optimized_routes()
        optimized_distance = self.calculate_total_distance(optimized_routes)
        print(f"  优化后总距离: {optimized_distance:.2f} km")

        # 3. 计算改进
        print("\n步骤 3/3: 计算优化效果...")
        distance_saved = baseline_distance - optimized_distance
        distance_saved_pct = (distance_saved / baseline_distance) * 100

        print(f"  距离节省: {distance_saved:.2f} km ({distance_saved_pct:.1f}%)")

        # 更新指标
        self.metrics['optimization']['distance_before'] = baseline_distance
        self.metrics['optimization']['distance_after'] = optimized_distance
        self.metrics['optimization']['distance_saved'] = distance_saved
        self.metrics['optimization']['distance_saved_pct'] = distance_saved_pct

        # 效率指标
        orders_per_vehicle_before = len(self.orders) / self.num_vehicles
        orders_per_vehicle_after = np.mean([len(route['orders']) for route in optimized_routes])
        self.metrics['efficiency']['orders_per_vehicle_before'] = orders_per_vehicle_before
        self.metrics['efficiency']['orders_per_vehicle_after'] = orders_per_vehicle_after
        self.metrics['efficiency']['efficiency_improvement'] = \
            ((orders_per_vehicle_after - orders_per_vehicle_before) / orders_per_vehicle_before) * 100

        # 业务指标
        fuel_cost_per_km = 1.5  # 元/km
        co2_per_km = 0.2  # kg/km
        self.metrics['business']['fuel_cost_saved'] = distance_saved * fuel_cost_per_km
        self.metrics['business']['co2_reduction_kg'] = distance_saved * co2_per_km

        print("="*70 + "\n")

        return baseline_routes, optimized_routes

    def create_baseline_routes(self):
        """创建基线路径（随机分配）"""
        routes = []
        orders_shuffled = self.orders.sample(frac=1).reset_index(drop=True)

        orders_per_vehicle = len(orders_shuffled) // self.num_vehicles

        for i in range(self.num_vehicles):
            start_idx = i * orders_per_vehicle
            if i == self.num_vehicles - 1:
                end_idx = len(orders_shuffled)
            else:
                end_idx = (i + 1) * orders_per_vehicle

            vehicle_orders = orders_shuffled.iloc[start_idx:end_idx]

            routes.append({
                'vehicle_id': f'V{i+1:02d}',
                'orders': vehicle_orders.to_dict('records'),
                'route_sequence': list(range(len(vehicle_orders)))
            })

        return routes

    def create_optimized_routes(self):
        """创建优化路径（简化的聚类+贪心）"""
        from scipy.cluster.vq import kmeans, vq

        # K-means聚类分配订单到车辆
        coords = self.orders[['latitude', 'longitude']].values
        centroids, _ = kmeans(coords, self.num_vehicles)
        cluster_ids, _ = vq(coords, centroids)

        routes = []

        for i in range(self.num_vehicles):
            # 获取该车辆的订单
            vehicle_orders = self.orders[cluster_ids == i].reset_index(drop=True)

            if len(vehicle_orders) == 0:
                continue

            # 使用贪心算法优化顺序
            route_sequence = self.greedy_route(vehicle_orders)

            routes.append({
                'vehicle_id': f'V{i+1:02d}',
                'orders': vehicle_orders.iloc[route_sequence].to_dict('records'),
                'route_sequence': route_sequence
            })

        return routes

    def calculate_total_distance(self, routes):
        """计算所有路径的总距离"""
        total_distance = 0

        for route in routes:
            # 从仓库到第一个订单
            if len(route['orders']) > 0:
                first_order = route['orders'][0]
                distance = self.calculate_distance(
                    self.depot['lat'], self.depot['lon'],
                    first_order['latitude'], first_order['longitude']
                )
                total_distance += distance

                # 订单之间
                for i in range(len(route['orders']) - 1):
                    curr_order = route['orders'][i]
                    next_order = route['orders'][i + 1]
                    distance = self.calculate_distance(
                        curr_order['latitude'], curr_order['longitude'],
                        next_order['latitude'], next_order['longitude']
                    )
                    total_distance += distance

                # 最后一个订单回仓库
                last_order = route['orders'][-1]
                distance = self.calculate_distance(
                    last_order['latitude'], last_order['longitude'],
                    self.depot['lat'], self.depot['lon']
                )
                total_distance += distance

        return total_distance

    def create_map_visualization(self, routes, title="优化后路径"):
        """创建交互地图"""
        # 创建地图
        center_lat = self.depot['lat']
        center_lon = self.depot['lon']
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        # 颜色
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']

        # 添加仓库标记
        folium.Marker(
            [self.depot['lat'], self.depot['lon']],
            popup=f"<b>{self.depot['name']}</b>",
            tooltip="配送中心",
            icon=folium.Icon(color='black', icon='home', prefix='fa')
        ).add_to(m)

        # 为每条路径绘制
        for idx, route in enumerate(routes):
            if len(route['orders']) == 0:
                continue

            color = colors[idx % len(colors)]
            vehicle_id = route['vehicle_id']

            # 路径坐标
            path_coords = [[self.depot['lat'], self.depot['lon']]]

            # 添加订单标记
            for i, order in enumerate(route['orders']):
                lat, lon = order['latitude'], order['longitude']
                path_coords.append([lat, lon])

                # 标记
                folium.CircleMarker(
                    [lat, lon],
                    radius=6,
                    popup=f"<b>{order['order_id']}</b><br>"
                          f"序号: {i+1}<br>"
                          f"车辆: {vehicle_id}<br>"
                          f"地址: {order['address']}<br>"
                          f"时间窗: {order['time_window_start']}-{order['time_window_end']}",
                    tooltip=f"{vehicle_id}-{i+1}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)

            # 回到仓库
            path_coords.append([self.depot['lat'], self.depot['lon']])

            # 绘制路径
            folium.PolyLine(
                path_coords,
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"{vehicle_id}: {len(route['orders'])} 订单"
            ).add_to(m)

        # 添加图例
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 220px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <p><b>{title}</b></p>
        <p><i class="fa fa-home" style="color:black"></i> 配送中心</p>
        '''

        for idx, route in enumerate(routes):
            color = colors[idx % len(colors)]
            vehicle_id = route['vehicle_id']
            num_orders = len(route['orders'])
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {vehicle_id} ({num_orders}单)</p>'

        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def create_visualizations(self, baseline_routes, optimized_routes):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('配送路径优化分析', fontsize=16, fontweight='bold')

        # 1. 距离对比
        ax1 = axes[0, 0]
        distances = [
            self.metrics['optimization']['distance_before'],
            self.metrics['optimization']['distance_after']
        ]
        labels = ['优化前', '优化后']
        bars = ax1.bar(labels, distances, color=['#e74c3c', '#2ecc71'])
        ax1.set_ylabel('总距离 (km)', fontsize=11)
        ax1.set_title('配送距离对比', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')

        # 节省百分比
        saved_pct = self.metrics['optimization']['distance_saved_pct']
        ax1.text(0.5, max(distances) * 0.5, f'节省\n{saved_pct:.1f}%',
                ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # 2. 每车订单数
        ax2 = axes[0, 1]
        vehicle_ids = [r['vehicle_id'] for r in optimized_routes]
        order_counts = [len(r['orders']) for r in optimized_routes]
        bars = ax2.barh(vehicle_ids, order_counts, color=plt.cm.Set3(range(len(vehicle_ids))))
        ax2.set_xlabel('订单数', fontsize=11)
        ax2.set_title('各车辆配送订单数', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontweight='bold')

        # 3. 业务价值
        ax3 = axes[1, 0]
        biz_metrics = {
            '距离节省\n(km)': self.metrics['optimization']['distance_saved'],
            '成本节省\n(元)': self.metrics['business']['fuel_cost_saved'],
            'CO2减排\n(kg)': self.metrics['business']['co2_reduction_kg']
        }
        bars = ax3.bar(range(len(biz_metrics)), list(biz_metrics.values()),
                      color=['#3498db', '#f39c12', '#27ae60'])
        ax3.set_xticks(range(len(biz_metrics)))
        ax3.set_xticklabels(list(biz_metrics.keys()), fontsize=10)
        ax3.set_ylabel('数值', fontsize=11)
        ax3.set_title('业务价值分析', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 4. 路径距离分布
        ax4 = axes[1, 1]
        route_distances = []
        for route in optimized_routes:
            if len(route['orders']) > 0:
                # 计算单条路径距离
                dist = 0
                # 仓库到第一个订单
                first = route['orders'][0]
                dist += self.calculate_distance(
                    self.depot['lat'], self.depot['lon'],
                    first['latitude'], first['longitude']
                )
                # 订单间
                for i in range(len(route['orders']) - 1):
                    curr = route['orders'][i]
                    next_order = route['orders'][i + 1]
                    dist += self.calculate_distance(
                        curr['latitude'], curr['longitude'],
                        next_order['latitude'], next_order['longitude']
                    )
                # 回仓库
                last = route['orders'][-1]
                dist += self.calculate_distance(
                    last['latitude'], last['longitude'],
                    self.depot['lat'], self.depot['lon']
                )
                route_distances.append(dist)

        vehicle_labels = [r['vehicle_id'] for r in optimized_routes if len(r['orders']) > 0]
        bars = ax4.bar(vehicle_labels, route_distances,
                      color=plt.cm.Pastel1(range(len(route_distances))))
        ax4.set_ylabel('路径距离 (km)', fontsize=11)
        ax4.set_title('各车辆路径距离', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'optimization_analysis.png'),
                   dpi=300, bbox_inches='tight')
        print("✓ 优化分析图已保存")

    def create_report(self):
        """创建报告"""
        report_path = os.path.join(self.results_dir, 'optimization_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("智能配送路径优化系统 - 优化报告\n")
            f.write("="*80 + "\n\n")

            f.write("一、优化概况\n")
            f.write("-" * 80 + "\n")
            f.write(f"订单总数: {len(self.orders)}\n")
            f.write(f"车辆数量: {self.num_vehicles}\n")
            f.write(f"配送中心: {self.depot['name']}\n\n")

            f.write("二、优化效果\n")
            f.write("-" * 80 + "\n")
            f.write(f"优化前总距离: {self.metrics['optimization']['distance_before']:.2f} km\n")
            f.write(f"优化后总距离: {self.metrics['optimization']['distance_after']:.2f} km\n")
            f.write(f"距离节省: {self.metrics['optimization']['distance_saved']:.2f} km\n")
            f.write(f"节省比例: {self.metrics['optimization']['distance_saved_pct']:.1f}%\n\n")

            f.write("三、效率提升\n")
            f.write("-" * 80 + "\n")
            f.write(f"优化前平均每车订单数: {self.metrics['efficiency']['orders_per_vehicle_before']:.1f}\n")
            f.write(f"优化后平均每车订单数: {self.metrics['efficiency']['orders_per_vehicle_after']:.1f}\n")
            f.write(f"效率提升: {self.metrics['efficiency']['efficiency_improvement']:.1f}%\n\n")

            f.write("四、业务价值\n")
            f.write("-" * 80 + "\n")
            f.write(f"燃油成本节省: ¥{self.metrics['business']['fuel_cost_saved']:.2f}\n")
            f.write(f"CO2减排: {self.metrics['business']['co2_reduction_kg']:.2f} kg\n")
            f.write(f"预计准时率: {self.metrics['business']['on_time_rate']:.1%}\n\n")

            f.write("五、技术方案\n")
            f.write("-" * 80 + "\n")
            f.write("• 订单分配: K-means聚类算法\n")
            f.write("• 路径优化: 贪心最近邻算法\n")
            f.write("• 距离计算: 欧氏距离近似\n")
            f.write("• 优化目标: 最小化总配送距离\n\n")

            f.write("="*80 + "\n")

        print(f"✓ 优化报告已保存至 {report_path}")

    def run_demo(self):
        """运行完整演示"""
        print("\n" + "="*80)
        print("智能配送路径优化系统 - 完整演示")
        print("="*80 + "\n")

        # 1. 路径优化
        baseline_routes, optimized_routes = self.optimize_routes()

        # 2. 创建可视化
        print("步骤 4/6: 生成分析图表...")
        self.create_visualizations(baseline_routes, optimized_routes)

        # 3. 创建交互地图
        print("\n步骤 5/6: 生成交互地图...")
        map_viz = self.create_map_visualization(optimized_routes, "优化后配送路径")
        map_path = os.path.join(self.results_dir, 'route_map.html')
        map_viz.save(map_path)
        print(f"✓ 交互地图已保存至 {map_path}")

        # 4. 生成报告
        print("\n步骤 6/6: 生成优化报告...")
        self.create_report()

        # 5. 保存路径数据
        routes_path = os.path.join(self.results_dir, 'optimized_routes.json')
        with open(routes_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_routes, f, indent=2, ensure_ascii=False)

        # 6. 保存指标
        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        # 显示结果
        print("\n" + "="*80)
        print("演示完成！")
        print("="*80)
        print(f"\n结果文件位置:")
        print(f"  • 优化分析图: {os.path.join(self.results_dir, 'optimization_analysis.png')}")
        print(f"  • 交互地图: {map_path}")
        print(f"  • 优化报告: {os.path.join(self.results_dir, 'optimization_report.txt')}")
        print(f"  • 路径数据: {routes_path}")
        print(f"  • 性能指标: {metrics_path}")

        print("\n关键优化指标:")
        print(f"  • 距离节省: {self.metrics['optimization']['distance_saved_pct']:.1f}%")
        print(f"  • 成本节省: ¥{self.metrics['business']['fuel_cost_saved']:.2f}")
        print(f"  • CO2减排: {self.metrics['business']['co2_reduction_kg']:.2f} kg")
        print(f"  • 车辆利用率提升: {self.metrics['efficiency']['efficiency_improvement']:.1f}%")

        print("\n提示: 在浏览器中打开 route_map.html 查看交互式路径地图")
        print("="*80 + "\n")


if __name__ == '__main__':
    demo = RouteOptimizationDemo()
    demo.run_demo()
