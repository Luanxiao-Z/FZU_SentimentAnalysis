import csv
import os
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

# 定义情感标签映射
EMOTION_MAPPING = {
    'like': {'coarse_label': 1, 'emotion_name': '开心', 'label': 1},
    'happiness': {'coarse_label': 1, 'emotion_name': '开心', 'label': 1},
    'disgust': {'coarse_label': 0, 'emotion_name': '厌恶', 'label': 5},
    'surprise': {'coarse_label': 2, 'emotion_name': '惊讶', 'label': 3},
    'sadness': {'coarse_label': 0, 'emotion_name': '悲伤', 'label': 0},
    'anger': {'coarse_label': 0, 'emotion_name': '生气', 'label': 2},
    'fear': {'coarse_label': 0, 'emotion_name': '恐惧', 'label': 4}
}


def validate_input_row(row):
    """验证输入行是否有效"""
    if len(row) < 2:
        return False
    if not row[0].strip() or not row[1].strip():
        return False
    return True


def process_file(input_path, output_path):
    processed_count = 0
    skipped_count = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8', newline='') as outfile:

            writer = csv.writer(outfile)
            writer.writerow(['coarse_label', 'emotion_name', 'label', 'text'])

            for line_num, line in enumerate(infile, 1):
                line = line.strip()

                # 去掉首尾的引号和空格
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1].strip()

                # 手动按第一个逗号分割（标签和文本）
                parts = line.split(',', 1)  # 只分割一次，保留文本中的逗号

                if len(parts) < 2:
                    skipped_count += 1
                    logging.warning(f"第 {line_num} 行格式错误，缺少逗号分隔: {line}")
                    continue

                label_part, text_part = parts[0].strip().lower(), parts[1].strip()

                # 验证非空
                if not label_part or not text_part:
                    skipped_count += 1
                    continue

                # 查找映射
                if label_part in EMOTION_MAPPING:
                    mapping = EMOTION_MAPPING[label_part]
                    writer.writerow([
                        mapping['coarse_label'],
                        mapping['emotion_name'],
                        mapping['label'],
                        text_part
                    ])
                    processed_count += 1
                else:
                    skipped_count += 1
                    logging.warning(f"第 {line_num} 行未知标签 '{label_part}'，原内容: {line}")

        logging.info(f"文件处理完成: {input_path} -> {output_path}")
        logging.info(f"处理行数: {processed_count}, 跳过行数: {skipped_count}")
        return True

    except Exception as e:
        logging.error(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def batch_process():
    """批量处理当前目录中的所有CSV文件"""
    # 获取当前目录
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, 'output')

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"创建输出目录: {output_dir}")

    processed_files = 0
    failed_files = 0

    # 遍历当前目录下的所有CSV文件
    for filename in os.listdir(current_dir):
        if filename.endswith('.csv') and not filename.startswith('processed_'):
            input_path = os.path.join(current_dir, filename)
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            if process_file(input_path, output_path):
                processed_files += 1
            else:
                failed_files += 1

    logging.info(f"批量处理完成: 成功 {processed_files} 个文件, 失败 {failed_files} 个文件")


if __name__ == "__main__":
    print("CSV 数据处理工具")
    print("正在处理当前目录下的所有CSV文件...")
    batch_process()
    print("处理完成！请查看 output 文件夹中的处理结果。")