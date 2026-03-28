#!/bin/bash

while true; do
    # 检查是否有 Train1201.py 的 python 进程
    RUNNING=$(ps aux | grep 'python.*Train_12_01.py' | grep -v grep)

    if [ -z "$RUNNING" ]; then
        echo "All Train1201.py processes finished. Starting Train1205.py..."
	python /ai/edallx/Graduate_Project_2025/NewMethods/Train_12_05.py
        break
    else
        echo "Still running Train1201.py, waiting..."
        sleep 300  # 每 5.0 min检查一次
    fi
done
