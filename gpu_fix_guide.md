# GPU修复指南 - RTX 2070 CUDA驱动问题

## 🔍 问题诊断

你的系统显示典型的**NVIDIA驱动版本不匹配**问题：

- **内核模块版本**: 580.65.06
- **NVML库版本**: 580.95  
- **已安装包版本**: 580.95.05

这种版本不一致导致`nvidia-smi`无法正常工作，进而影响PyTorch的CUDA检测。

## 🛠️ 修复方案

### 方案1: 降级到稳定驱动版本（推荐）

对于RTX 2070，建议使用更稳定的驱动版本：

```bash
# 1. 卸载当前驱动
sudo apt remove --purge nvidia-*
sudo apt autoremove

# 2. 安装稳定版本（535系列）
sudo apt update
sudo apt install nvidia-driver-535 nvidia-dkms-535

# 3. 重启系统
sudo reboot
```

### 方案2: 重新安装匹配的580驱动

如果坚持使用最新版本：

```bash
# 1. 完全卸载当前驱动
sudo apt remove --purge nvidia-* libnvidia-*
sudo apt autoremove
sudo apt clean

# 2. 重新安装580系列（确保版本一致）
sudo apt install nvidia-driver-580-open nvidia-dkms-580-open

# 3. 重启
sudo reboot
```

### 方案3: 手动安装特定版本

```bash
# 1. 查看可用版本
ubuntu-drivers devices

# 2. 安装特定版本（例如570系列）
sudo apt install nvidia-driver-570 nvidia-dkms-570

# 3. 重启
sudo reboot
```

## 🔧 详细步骤

### 步骤1: 备份和清理

```bash
# 创建系统还原点（可选但推荐）
sudo timeshift --create --comments "Before NVIDIA driver fix"

# 停止显示管理器
sudo systemctl stop gdm3  # 或 lightdm, sddm 取决于你的系统

# 卸载当前驱动
sudo apt remove --purge '^nvidia-.*'
sudo apt autoremove
sudo apt clean

# 删除残留配置
sudo rm -rf /etc/nvidia*
sudo rm -rf /usr/share/nvidia*
```

### 步骤2: 安装推荐驱动

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装推荐版本（535系列 - 非常稳定）
sudo apt install -y nvidia-driver-535 nvidia-dkms-535 nvidia-utils-535

# 或者安装570系列（较新但仍稳定）
# sudo apt install -y nvidia-driver-570 nvidia-dkms-570 nvidia-utils-570
```

### 步骤3: 验证安装

```bash
# 重启后检查
nvidia-smi
nvidia-settings

# 检查内核模块
lsmod | grep nvidia

# 检查驱动版本
cat /proc/driver/nvidia/version
```

## 🧪 PyTorch兼容性测试

修复驱动后，测试PyTorch：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 测试CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 测试GPU计算
python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)
    print('GPU computation successful!')
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('CUDA not available - check driver installation')
"
```

## 🎯 针对RTX 2070的特别建议

### 最佳驱动版本
- **535系列**: 最稳定，兼容性最好
- **470系列**: 经典长期支持版本
- **570系列**: 较新但稳定
- **避免580系列**: 太新可能有兼容性问题

### CUDA版本匹配
- **CUDA 11.8**: 推荐（PyTorch已安装此版本）
- **CUDA 12.1**: 可选（需要重新安装PyTorch）
- **避免CUDA 12.2+**: 可能有兼容性问题

## 🚨 常见问题解决

### 问题1: 黑屏或无法启动
```bash
# 进入TTY模式 (Ctrl+Alt+F3)
# 卸载驱动
sudo apt remove --purge nvidia-*
# 重装开源驱动
sudo apt install xserver-xorg-video-nouveau
sudo reboot
```

### 问题2: 内核模块未加载
```bash
# 检查内核版本
uname -r

# 重新编译DKMS模块
sudo dkms autoinstall
sudo update-initramfs -u
```

### 问题3: 版本冲突
```bash
# 检查所有NVIDIA包
 dpkg -l | grep nvidia

# 如果有冲突，统一版本
sudo apt install nvidia-driver-535 libnvidia-gl-535 nvidia-utils-535
```

## 📋 推荐操作流程

1. **备份重要数据**
2. **选择降级到535系列**（最稳定）
3. **完全卸载当前驱动**
4. **安装推荐版本**
5. **重启并验证**
6. **测试PyTorch CUDA**

## ✋ 如果出现问题

如果修复过程中遇到问题：

1. **不要panic** - 可以回到CPU模式学习
2. **记录错误信息** - 便于排查
3. **尝试安全模式** - 开机时选择恢复模式
4. **寻求社区帮助** - Ubuntu论坛、NVIDIA开发者论坛

记住：**CPU模式完全可以学习LLM的所有概念**，只是训练速度慢一些。修复GPU是为了更好的性能，但不是学习的障碍！

## 🔗 有用链接

- [NVIDIA驱动下载](https://www.nvidia.com/Download/index.aspx)
- [Ubuntu NVIDIA文档](https://ubuntu.com/server/docs/nvidia-drivers)
- [PyTorch CUDA支持](https://pytorch.org/get-started/locally/)
- [NVIDIA开发者论坛](https://forums.developer.nvidia.com/)