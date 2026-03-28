#!/bin/bash

# 检查输入参数
if [[ $# -ne 2 ]]; then
    echo "用法: $0 <源目录> <目标目录>"
    exit 1
fi

# 读取用户输入的源目录和目标目录
SOURCE_DIR="$1"
DEST_DIR="$2"

# 确保源目录存在
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "错误: 源目录 '$SOURCE_DIR' 不存在！"
    exit 1
fi

# 确保目标目录存在
mkdir -p "$DEST_DIR"

# 获取源目录中的所有文件，并随机选取 9600 个
FILES=($(find "$SOURCE_DIR" -type f | shuf))
TOTAL_FILES=${#FILES[@]}
COPY_COUNT=$((TOTAL_FILES < 9600 ? TOTAL_FILES : 9600))  # 取 9600 或最大数量
COUNT=0

# 进度更新函数
progress() {
    COUNT=$((COUNT + 1))
    echo -ne "进度: $COUNT / $COPY_COUNT 文件已复制\r"
}

# 遍历选取的文件并复制
for ((i = 0; i < COPY_COUNT; i++)); do
    FILE="${FILES[$i]}"
    
    # 获取文件名（去掉路径）
    BASENAME=$(basename "$FILE")

    # 复制文件（直接放入目标目录）
    cp "$FILE" "$DEST_DIR/$BASENAME"

    # 更新进度
    progress
done

echo -e "\n? 已随机复制 $COPY_COUNT 个文件到 '$DEST_DIR'！"
