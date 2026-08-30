#!/bin/bash
echo "Killing all python processes..."
pkill -9 -f python3 || true
sleep 8
echo "GPU status:"
nvidia-smi
echo "Free memory check done"
