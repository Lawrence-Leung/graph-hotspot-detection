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

# 获取源目录中的所有文件
FILES=($(find "$SOURCE_DIR" -type f))
TOTAL_FILES=${#FILES[@]}
COUNT=0

# 进度更新函数
progress() {
    COUNT=$((COUNT + 1))
    echo -ne "进度: $COUNT / $TOTAL_FILES 文件已移动\r"
}

# 遍历所有文件并移动
for FILE in "${FILES[@]}"; do
    # 获取文件名（去掉路径）
    BASENAME=$(basename "$FILE")

    # 移动文件（直接放入目标目录）
    mv "$FILE" "$DEST_DIR/$BASENAME"

    # 更新进度
    progress
done

echo -e "\n? 已成功移动 $TOTAL_FILES 个文件到 '$DEST_DIR'！"
