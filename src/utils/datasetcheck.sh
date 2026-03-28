#!/bin/bash

dir=$1
if [ -z "$dir" ]; then
  echo "用法: $0 <目标文件夹路径>"
  exit 1
fi

count0=0
count1=0
files=($dir/*.pt)
total=${#files[@]}
current=0

for file in "${files[@]}"; do
  current=$((current + 1))
    # 提取文件名
    filename=$(basename "$ptfile")

    # 使用正则提取 label
    if [[ $filename =~ _GphLst_([01])\.pt$ ]]; then
        label=${BASH_REMATCH[1]}
        if [ "$label" -eq 0 ]; then
            ((count_0++))
        elif [ "$label" -eq 1 ]; then
            ((count_1++))
        fi
    fi

  # 输出进度条
  echo -ne "\r已处理: $current/$total 个文件"
done

echo -e "\n统计完成:"
echo "label=0 的文件数: $count0"
echo "label=1 的文件数: $count1"
