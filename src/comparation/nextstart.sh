#!/bin/bash

while true; do
    # 检查是否有 BatchGraphGenerator05.py 的 python 进程
    RUNNING=$(ps aux | grep 'python.*Test.py' | grep -v grep)

    if [ -z "$RUNNING" ]; then
        echo "All Test.py (19) processes finished. Starting Test.py (12)..."
	python /ai/edallx/Graduate_Project_2025/Geng2020Code/Test.py
        break
    else
        echo "Still running Test.py, waiting..."
        sleep 300  # 每 5.0 min检查一次
    fi
done
