import tensorflow as tf

# 指定 .tfevents 文件的路径
file_path = "./events.out.tfevents.1695064972.nid001236.575982.0"

# 使用 summary_iterator 迭代 .tfevents 文件
for summary in tf.compat.v1.train.summary_iterator(file_path):
    # 获取当前事件的步数（step）
    step = summary.step
    
    # 遍历当前事件中的所有摘要（summary）
    for value in summary.summary.value:
        # 获取摘要的标签（tag）和简单值（simple_value）
        tag = value.tag
        simple_value = value.simple_value
        
        # 打印摘要信息
        print(f"Step: {step}, Tag: {tag}, SimpleValue: {simple_value}")

