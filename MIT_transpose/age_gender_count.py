import os
import csv
import wfdb

def parse_hea_file(hea_path):
    """使用wfdb标准方式解析本地的hea文件，提取年龄和性别信息"""
    record_name = os.path.splitext(os.path.basename(hea_path))[0]
    # 添加rdheader的force_local=True参数
    header = wfdb.rdheader(hea_path)
    age = header.comments.get('age') if hasattr(header.comments, 'get') else None
    sex = header.comments.get('sex') if hasattr(header.comments, 'get') else None
    # 有些版本的wfdb，age/sex直接是属性
    if age is None and hasattr(header, 'age'):
        age = header.age
    if sex is None and hasattr(header, 'sex'):
        sex = header.sex
    # 有些header.comments是list
    if isinstance(header.comments, list):
        for c in header.comments:
            if 'Age' in c:
                age = c.split(':')[-1].strip()
            if 'Sex' in c:
                sex = c.split(':')[-1].strip()
    return age, sex

def main():
    data_dir = '/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/ori_data'
    output_csv = '/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/age_gender_stat.csv'
    results = []

    for file in os.listdir(data_dir):
        if file.endswith('.hea'):
            print(f'正在处理: {file}')  # 新增：输出当前处理的文件名
            patient_id = file[:-4]
            hea_path = os.path.join(data_dir, patient_id)
            age, sex = parse_hea_file(hea_path)
            results.append({'patient_id': patient_id, 'age': age, 'sex': sex})

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['patient_id', 'age', 'sex']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f'统计完成，结果已写入 {output_csv}')

if __name__ == '__main__':
    main()