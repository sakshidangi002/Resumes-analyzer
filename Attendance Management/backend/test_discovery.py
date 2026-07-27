"""
Test script for DVR discovery without API authentication.
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.services.hikvision_discovery import discover_cameras

# Test DVR discovery
ip = os.getenv("DVR_IP", "")
port = int(os.getenv("DVR_PORT", "8000"))
username = os.getenv("DVR_USERNAME", "")
password = os.getenv("DVR_PASSWORD", "")

print(f"Testing DVR discovery for {ip}:{port}")
print(f"Username: {username}")
print(f"Password: {password}")
print()

success, discovered, error = discover_cameras(ip, port, username, password)

if success:
    print("✓ Discovery successful!")
    print(f"\nDevice Information:")
    print(f"  Model: {discovered.model}")
    print(f"  Firmware: {discovered.firmware}")
    print(f"  Serial: {discovered.serial}")
    print(f"  Total Channels: {discovered.total_channels}")
    print(f"  Analog Channels: {discovered.analog_channels}")
    print(f"  IP Channels: {discovered.ip_channels}")
    print(f"\nDiscovered Channels:")
    print("  IMPORTANT: Only select channels that show 'online' status")
    for channel in discovered.channels:
        status_indicator = "✓" if channel.status == "online" else "✗"
        print(f"  {status_indicator} Channel {channel.id}: {channel.name} ({channel.status}, {channel.channel_type})")
else:
    print(f"✗ Discovery failed: {error}")
    sys.exit(1)
