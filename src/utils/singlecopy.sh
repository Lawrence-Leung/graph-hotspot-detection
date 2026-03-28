#!/bin/bash

# 检查参数
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

# 计算文件总数（用于进度显示）
TOTAL_FILES=$(find "$SOURCE_DIR" -type f | wc -l)
COUNT=0

# 进度更新函数
progress() {
    COUNT=$((COUNT + 1))
    echo -ne "进度: $COUNT / $TOTAL_FILES 文件已复制\r"
}

# 遍历文件并逐个复制
find "$SOURCE_DIR" -type f | while IFS= read -r FILE; do
    # 获取文件名（去掉路径）
    BASENAME=$(basename "$FILE")

    # 执行复制（放入目标目录，不保留原目录结构）
    cp "$FILE" "$DEST_DIR/$BASENAME"

    # 更新进度
    progress
done

echo -e "\n? 所有文件已成功复制到 '$DEST_DIR'！"
