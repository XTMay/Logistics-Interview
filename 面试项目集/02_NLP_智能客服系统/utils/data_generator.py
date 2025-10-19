"""
数据生成器 - 生成物流客服对话数据和知识库
用于演示目的
"""

import json
import os
import random
from datetime import datetime, timedelta


class CustomerServiceDataGenerator:
    """物流客服数据生成器"""

    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        self.kb_dir = os.path.join(output_dir, 'knowledge_base')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.kb_dir, exist_ok=True)

        # 意图类别
        self.intents = [
            'order_query',      # 订单查询
            'tracking',         # 物流追踪
            'complaint',        # 投诉
            'address_change',   # 修改地址
            'refund',          # 退款
            'pricing',         # 价格查询
            'delivery_time',   # 配送时间
            'package_info'     # 包裹信息
        ]

        # 情感标签
        self.sentiments = ['positive', 'neutral', 'negative', 'urgent']

        # 模板数据
        self.templates = {
            'order_query': [
                '我的订单{order_id}现在是什么状态？',
                '帮我查一下订单{order_id}',
                '订单{order_id}什么时候发货？',
                '查询订单编号{order_id}的信息',
            ],
            'tracking': [
                '快递单号{tracking_id}在哪里了？',
                '我的包裹{tracking_id}到哪了？',
                '查询物流{tracking_id}',
                '追踪快递{tracking_id}的位置',
            ],
            'complaint': [
                '我的包裹{tracking_id}怎么还没到？已经{days}天了！',
                '配送员态度很差，投诉！',
                '包裹{tracking_id}破损了，怎么处理？',
                '快递迟迟不到，要求赔偿！',
            ],
            'address_change': [
                '订单{order_id}能修改收货地址吗？',
                '我想改一下配送地址',
                '地址填错了，能改成{address}吗？',
                '订单{order_id}地址修改',
            ],
            'refund': [
                '订单{order_id}申请退款',
                '不想要了，怎么退款？',
                '退款流程是什么？',
                '订单{order_id}退款进度',
            ],
            'pricing': [
                '从{city1}到{city2}运费多少？',
                '寄{weight}kg的包裹多少钱？',
                '价格表在哪里看？',
                '有什么优惠活动吗？',
            ],
            'delivery_time': [
                '从{city1}到{city2}要多久？',
                '什么时候能送到？',
                '{city}的配送时间是几点到几点？',
                '加急最快多久？',
            ],
            'package_info': [
                '可以寄{item}吗？',
                '{item}属于违禁品吗？',
                '包裹尺寸限制是多少？',
                '最大重量能寄多少？',
            ]
        }

        # 示例数据
        self.order_ids = [f'ORD{random.randint(10000000, 99999999)}' for _ in range(20)]
        self.tracking_ids = [f'SF{random.randint(1000000000, 9999999999)}' for _ in range(20)]
        self.cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安']
        self.addresses = ['朝阳区建国路1号', '浦东新区陆家嘴', '天河区珠江新城', '南山区科技园']
        self.items = ['文件', '电子产品', '衣物', '食品', '书籍', '化妆品']

    def generate_query(self, intent):
        """生成单个查询"""
        template = random.choice(self.templates[intent])

        # 填充变量
        query = template.format(
            order_id=random.choice(self.order_ids),
            tracking_id=random.choice(self.tracking_ids),
            days=random.randint(3, 10),
            address=random.choice(self.addresses),
            city1=random.choice(self.cities),
            city2=random.choice([c for c in self.cities if c != random.choice(self.cities)]),
            city=random.choice(self.cities),
            weight=random.randint(1, 30),
            item=random.choice(self.items)
        )

        # 分配情感
        if intent == 'complaint':
            sentiment = random.choice(['negative', 'urgent'])
        elif intent == 'order_query':
            sentiment = random.choice(['neutral', 'positive'])
        else:
            sentiment = random.choice(['neutral', 'neutral', 'positive'])

        return query, intent, sentiment

    def generate_training_data(self, num_samples=500):
        """生成训练数据"""
        print(f"生成 {num_samples} 条训练数据...")

        data = []
        for _ in range(num_samples):
            intent = random.choice(self.intents)
            query, intent_label, sentiment = self.generate_query(intent)

            data.append({
                'query': query,
                'intent': intent_label,
                'sentiment': sentiment
            })

        # 保存
        output_path = os.path.join(self.output_dir, 'training_data.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ 训练数据已保存至 {output_path}")
        return data

    def generate_faq(self):
        """生成常见问题FAQ"""
        faqs = [
            {
                'question': '如何查询订单状态？',
                'answer': '您可以通过以下方式查询订单状态：\n1. 在我们的官网或APP中输入订单号查询\n2. 联系客服提供订单号\n3. 查看订单确认邮件中的追踪链接',
                'category': 'order_query',
                'keywords': ['订单', '查询', '状态']
            },
            {
                'question': '物流一般需要多长时间？',
                'answer': '配送时效因地区而异：\n- 同城：1-2天\n- 省内：2-3天\n- 跨省：3-5天\n- 偏远地区：5-7天\n加急服务可缩短1-2天',
                'category': 'delivery_time',
                'keywords': ['物流', '时间', '多久', '配送']
            },
            {
                'question': '如何修改收货地址？',
                'answer': '订单发货前可以修改地址：\n1. 登录账户，在订单详情中选择"修改地址"\n2. 联系客服协助修改\n注意：订单已发货后无法修改，但可以联系配送员协商',
                'category': 'address_change',
                'keywords': ['修改', '地址', '收货']
            },
            {
                'question': '运费如何计算？',
                'answer': '运费计算基于以下因素：\n1. 重量：首重1kg起\n2. 距离：同城、省内、跨省\n3. 尺寸：超大件另计\n4. 服务类型：标准/加急\n具体价格请使用运费计算器或咨询客服',
                'category': 'pricing',
                'keywords': ['运费', '价格', '计算', '多少钱']
            },
            {
                'question': '包裹破损怎么办？',
                'answer': '如收到破损包裹：\n1. 签收时当场拍照记录\n2. 联系客服提供照片和订单号\n3. 我们将在24小时内处理\n4. 根据情况提供换货或赔偿\n建议：购买保价服务可获得更好保障',
                'category': 'complaint',
                'keywords': ['破损', '损坏', '赔偿']
            },
            {
                'question': '可以寄送哪些物品？',
                'answer': '可寄送物品：\n✓ 文件、书籍、衣物\n✓ 电子产品（需包装完好）\n✓ 非易碎日用品\n\n禁运物品：\n✗ 易燃易爆物品\n✗ 有毒有害物质\n✗ 违禁品\n✗ 活体动物\n详细清单请查看官网',
                'category': 'package_info',
                'keywords': ['寄送', '物品', '禁运', '违禁品']
            },
            {
                'question': '如何申请退款？',
                'answer': '退款流程：\n1. 登录账户，进入订单详情\n2. 选择"申请退款"\n3. 填写退款原因\n4. 提交申请\n5. 客服审核（1-3个工作日）\n6. 审核通过后3-7个工作日到账\n\n注意：已发货订单需先拒收或退回',
                'category': 'refund',
                'keywords': ['退款', '退货', '申请']
            },
            {
                'question': '快递一直显示在途中怎么办？',
                'answer': '如物流长时间未更新：\n1. 查看最后更新时间\n2. 如超过48小时未更新，联系客服查询\n3. 我们将联系配送网点核实\n4. 如确认遗失，将启动赔付流程\n\n可能原因：网点中转延迟、扫描遗漏、系统更新延迟',
                'category': 'tracking',
                'keywords': ['在途', '未更新', '物流', '停滞']
            }
        ]

        output_path = os.path.join(self.output_dir, 'faq.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(faqs, f, indent=2, ensure_ascii=False)

        print(f"✓ FAQ已保存至 {output_path}")
        return faqs

    def generate_knowledge_base(self):
        """生成知识库文档"""
        documents = {
            '配送政策.txt': """物流配送政策

一、配送时效
1. 同城配送：1-2个工作日
2. 省内配送：2-3个工作日
3. 跨省配送：3-5个工作日
4. 偏远地区：5-7个工作日

二、配送时间
每日配送时间：9:00-18:00
周末及法定节假日正常配送

三、配送范围
覆盖全国主要城市及县级地区
部分偏远地区可能无法配送，请咨询客服

四、签收要求
1. 收件人本人签收
2. 代收需提供收件人授权
3. 签收时请检查包裹完整性
4. 如有破损请当场拍照并拒收

五、特殊情况处理
- 收件人不在：配送员会电话联系，协商配送时间
- 地址错误：联系客服修改或退回
- 无人签收：存放至快递柜或代收点
""",

            '收费标准.txt': """物流收费标准

一、基础运费
1. 同城配送
   - 首重（1kg以内）：8元
   - 续重（每增加1kg）：2元

2. 省内配送
   - 首重（1kg以内）：10元
   - 续重（每增加1kg）：3元

3. 跨省配送
   - 首重（1kg以内）：15元
   - 续重（每增加1kg）：5元

二、增值服务
1. 加急服务：+30%运费
2. 保价服务：按声明价值0.5%收取
3. 代收货款：按代收金额1%收取
4. 签收回单：5元/份

三、优惠活动
1. 月结客户：享受9折优惠
2. 批量寄送（>10件）：8.5折
3. 新用户首单免运费

四、重量计算
按实际重量和体积重量取较大值
体积重量 = 长×宽×高(cm) ÷ 6000
""",

            '退款政策.txt': """退款政策

一、退款条件
1. 未发货订单：全额退款
2. 已发货未签收：拒收后全额退款
3. 已签收：根据具体情况处理

二、退款流程
1. 提交退款申请
2. 客服审核（1-3个工作日）
3. 审核通过后退款
4. 退款到账时间：
   - 原路退回：3-7个工作日
   - 银行转账：1-3个工作日

三、不予退款情况
1. 签收超过7天
2. 包裹已被使用或损坏
3. 特价商品（明确标注不退款）
4. 定制商品

四、运费处理
1. 我方原因：全额退还运费
2. 客户原因：扣除实际产生的运费

五、赔偿说明
1. 包裹遗失：按声明价值或实际价值赔偿（最高2000元）
2. 包裹破损：按损坏程度赔偿
3. 延误：根据保价协议赔偿
4. 保价包裹：按保价金额赔偿
""",

            '禁运物品清单.txt': """禁运物品清单

一、绝对禁运物品
1. 枪支弹药、爆炸物品
2. 易燃易爆物品（汽油、烟花爆竹等）
3. 腐蚀性物品（强酸、强碱等）
4. 毒品及吸毒工具
5. 放射性物品
6. 传染病病原体
7. 国家法律法规禁止流通的物品

二、限制寄递物品
1. 活体动植物（需特殊许可）
2. 液体（需特殊包装）
3. 粉末状物品（需说明）
4. 电池（限定类型和数量）
5. 刀具（需符合规定）

三、需要特殊处理物品
1. 电子产品：需包装完好，防震保护
2. 玻璃陶瓷：易碎品标识，加固包装
3. 食品：保质期内，密封包装
4. 文件：防水包装
5. 贵重物品：建议保价

四、包装要求
1. 外包装完整、牢固
2. 内部物品固定，防止移动
3. 防震、防水措施
4. 标识清晰（易碎、向上等）

五、违规处理
寄送禁运物品将：
1. 拒绝收寄
2. 通报相关部门
3. 追究法律责任
"""
        }

        for filename, content in documents.items():
            filepath = os.path.join(self.kb_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        print(f"✓ 知识库文档已保存至 {self.kb_dir}")
        return list(documents.keys())

    def generate_all(self):
        """生成所有数据"""
        print("\n" + "="*70)
        print("物流客服数据生成")
        print("="*70 + "\n")

        # 1. 生成训练数据
        training_data = self.generate_training_data(num_samples=500)

        # 2. 生成FAQ
        faqs = self.generate_faq()

        # 3. 生成知识库
        kb_docs = self.generate_knowledge_base()

        # 统计
        intent_dist = {}
        sentiment_dist = {}

        for item in training_data:
            intent = item['intent']
            sentiment = item['sentiment']
            intent_dist[intent] = intent_dist.get(intent, 0) + 1
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

        print("\n" + "="*70)
        print("数据生成完成")
        print("="*70)
        print(f"训练数据: {len(training_data)} 条")
        print(f"FAQ: {len(faqs)} 条")
        print(f"知识库文档: {len(kb_docs)} 个")

        print(f"\n意图分布:")
        for intent, count in sorted(intent_dist.items()):
            print(f"  {intent}: {count} ({count/len(training_data)*100:.1f}%)")

        print(f"\n情感分布:")
        for sentiment, count in sorted(sentiment_dist.items()):
            print(f"  {sentiment}: {count} ({count/len(training_data)*100:.1f}%)")

        print("="*70 + "\n")


if __name__ == '__main__':
    generator = CustomerServiceDataGenerator()
    generator.generate_all()
