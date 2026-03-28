#!/bin/bash

# 目标文件夹（如果未提供，则使用当前目录）
TARGET_DIR="${1:-.}"

# 计数器存储文件
HS_INDEX_FILE="$TARGET_DIR/.hs_index"
NHS_INDEX_FILE="$TARGET_DIR/.nhs_index"

# 初始化计数器（如果文件不存在则创建）
if [[ ! -f "$HS_INDEX_FILE" ]]; then echo 0 > "$HS_INDEX_FILE"; fi
if [[ ! -f "$NHS_INDEX_FILE" ]]; then echo 0 > "$NHS_INDEX_FILE"; fi

# 遍历所有文件
find "$TARGET_DIR" -type f | while IFS= read -r FILE; do
    DIR=$(dirname "$FILE")
    BASENAME=$(basename "$FILE")
    EXT="${BASENAME##*.}"
    LOCK_FILE="$DIR/.rename_lock"

    # 处理 nonhotspot 文件
    if [[ "$BASENAME" == *"nonhotspot"* ]]; then
        # 获取并递增编号
        exec 200>"$LOCK_FILE"  # 获取锁
        flock -x 200
        NHS_INDEX=$(($(cat "$DIR/.nhs_index") + 1))
        echo "$NHS_INDEX" > "$DIR/.nhs_index"
        flock -u 200  # 释放锁
        exec 200>&-

        NEW_NAME=$(printf "NHS%03d.%s" "$NHS_INDEX" "$EXT")

    # 处理 hotspot 文件
    elif [[ "$BASENAME" == *"hotspot"* ]]; then
        # 获取并递增编号
        exec 200>"$LOCK_FILE"  # 获取锁
        flock -x 200
        HS_INDEX=$(($(cat "$DIR/.hs_index") + 1))
        echo "$HS_INDEX" > "$DIR/.hs_index"
        flock -u 200  # 释放锁
        exec 200>&-

        NEW_NAME=$(printf "HS%03d.%s" "$HS_INDEX" "$EXT")
    else
        continue  # 跳过不符合条件的文件
    fi

    # 确保不会覆盖已有文件
    NEW_PATH="$DIR/$NEW_NAME"
    while [[ -e "$NEW_PATH" ]]; do
        if [[ "$BASENAME" == *"nonhotspot"* ]]; then
            NHS_INDEX=$((NHS_INDEX + 1))
            echo "$NHS_INDEX" > "$DIR/.nhs_index"
            NEW_NAME=$(printf "NHS%03d.%s" "$NHS_INDEX" "$EXT")
        elif [[ "$BASENAME" == *"hotspot"* ]]; then
            HS_INDEX=$((HS_INDEX + 1))
            echo "$HS_INDEX" > "$DIR/.hs_index"
            NEW_NAME=$(printf "HS%03d.%s" "$HS_INDEX" "$EXT")
        fi
        NEW_PATH="$DIR/$NEW_NAME"
    done

    # 执行重命名
    mv -v "$FILE" "$NEW_PATH"
    echo "已重命名: $BASENAME -> $NEW_NAME"

done

echo "所有文件处理完成！"
