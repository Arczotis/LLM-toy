#!/bin/bash
# Quick GPU fix script for RTX 2070

echo "🚀 RTX 2070 GPU Driver Fix Script"
echo "=================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (sudo)"
   exit 1
fi

echo "📋 Current Status:"
echo "NVIDIA Driver Version:"
cat /proc/driver/nvidia/version 2>/dev/null || echo "Driver info not available"

echo ""
echo "🔧 Starting driver cleanup..."

# Step 1: Stop display manager
echo "1. Stopping display manager..."
systemctl stop gdm3 2>/dev/null || systemctl stop lightdm 2>/dev/null || systemctl stop sddm 2>/dev/null

# Step 2: Uninstall current NVIDIA drivers
echo "2. Removing current NVIDIA drivers..."
apt remove --purge -y '^nvidia-.*' libnvidia-*
apt autoremove -y
apt clean

# Step 3: Remove residual files
echo "3. Cleaning residual files..."
rm -rf /etc/nvidia* /usr/share/nvidia* /usr/lib/x86_64-linux-gnu/nvidia*

# Step 4: Update system
echo "4. Updating system..."
apt update
apt upgrade -y

echo ""
echo "📋 Available NVIDIA drivers:"
ubuntu-drivers devices | grep -A 20 "GeForce RTX 2070"

echo ""
echo "🎯 Choose installation option:"
echo "1) Install stable 535 series (RECOMMENDED)"
echo "2) Install 570 series (newer but stable)"
echo "3) Install 470 series (LTS version)"
echo "4) Skip installation (manual install later)"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Installing NVIDIA driver 535 series..."
        apt install -y nvidia-driver-535 nvidia-dkms-535 nvidia-utils-535
        ;;
    2)
        echo "Installing NVIDIA driver 570 series..."
        apt install -y nvidia-driver-570 nvidia-dkms-570 nvidia-utils-570
        ;;
    3)
        echo "Installing NVIDIA driver 470 series..."
        apt install -y nvidia-driver-470 nvidia-dkms-470 nvidia-utils-470
        ;;
    4)
        echo "Skipping driver installation. You can install manually later."
        echo "Recommended: sudo apt install nvidia-driver-535"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Step 5: Update initramfs
echo "5. Updating initramfs..."
update-initramfs -u

echo ""
echo "✅ Driver installation complete!"
echo ""
echo "🔄 System will reboot in 10 seconds..."
echo "Press Ctrl+C to cancel reboot and reboot manually later"

sleep 10
