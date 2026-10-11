"""DetectionBench —— 面向小目标检测的可复现评测框架。

模块划分::

    core.io         统一数据接口（COCO / YOLO / VOC 三种标注格式 + 图像属性元数据）
    core.metrics    分层 AP 计算（按目标尺寸/距离/模糊/光照分组评测）
    core.eval       COCO 标准评测 + 评测循环
    core.speed      FPS / 延迟 / 峰值显存测量
    core.results    实验记录（result.json）+ 一键汇总为论文表格
    core.registry   模型适配器注册表
    models/         各检测框架的适配器实现
    tools/          CLI 入口

设计原则（这三条是这个框架存在的理由）::

    1. 统一预处理 —— 所有模型走同一套 letterbox + 归一化，绝不让各模型自己处理。
    2. 统一评测 —— 所有模型走同一个评测脚本、同一套指标定义。
    3. 全部落盘 —— 每次实验自动记录 git commit / 配置 / 随机种子 / 环境版本，
                    三个月后你还能知道某个数字是怎么来的。
"""

__version__ = "0.1.0"
