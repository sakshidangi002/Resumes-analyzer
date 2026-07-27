"""
hikvision_discovery.py
======================
Device discovery service for Hikvision DVRs using HCNetSDK.

This module provides functionality to discover cameras connected to a Hikvision DVR
by logging in via HCNetSDK and retrieving channel information.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.services.hcnetsdk_wrapper import (
    _sdk_wrapper,
    NET_DVR_DEVICEINFO,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredChannel:
    """Represents a discovered camera channel."""
    id: int
    name: str
    status: str  # "online" or "offline"
    channel_type: str  # "analog" or "ip"
    resolution: Optional[str] = None


@dataclass
class DiscoveredDevice:
    """Represents a discovered Hikvision DVR device."""
    model: str
    firmware: str
    serial: str
    total_channels: int
    analog_channels: int
    ip_channels: int
    channels: list[DiscoveredChannel]


class HikvisionDiscoveryService:
    """Service for discovering cameras on Hikvision DVRs."""
    
    def __init__(self):
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Ensure HCNetSDK is initialized."""
        if not self._initialized:
            if not _sdk_wrapper.initialize():
                logger.error("Failed to initialize HCNetSDK for discovery")
                return False
            self._initialized = True
        return True
    
    def discover(
        self,
        ip: str,
        port: int,
        username: str,
        password: str
    ) -> tuple[bool, Optional[DiscoveredDevice], str]:
        """
        Discover cameras on a Hikvision DVR.
        
        Args:
            ip: DVR IP address
            port: DVR port (default 8000)
            username: DVR username
            password: DVR password
            
        Returns:
            Tuple of (success, discovered_device, error_message)
        """
        if not self._ensure_initialized():
            return False, None, "Failed to initialize HCNetSDK"
        
        # Login to DVR
        success, login_id, device_info = _sdk_wrapper.login(ip, port, username, password)
        if not success:
            return False, None, f"Login failed: {_sdk_wrapper.get_last_error()}"
        
        try:
            # Get device configuration
            device_config = self._get_device_config(login_id)
            if not device_config:
                return False, None, "Failed to retrieve device configuration"
            
            # Discover channels
            channels = self._discover_channels(login_id, device_info)
            
            # Create discovered device
            discovered = DiscoveredDevice(
                model=device_config.get("model", "Unknown"),
                firmware=device_config.get("firmware", "Unknown"),
                serial=device_info.sSerialNumber.decode('ascii', errors='ignore').rstrip('\x00'),
                total_channels=device_info.byChanNum,
                analog_channels=device_info.byChanNum,  # Simplified
                ip_channels=0,  # Will be calculated from channel discovery
                channels=channels
            )
            
            logger.info(
                f"Discovered {len(channels)} channels on {ip}:{port} "
                f"(model: {discovered.model}, serial: {discovered.serial})"
            )
            
            return True, discovered, ""
            
        finally:
            # Always logout
            _sdk_wrapper.logout(login_id)
    
    def _get_device_config(self, login_id: int) -> Optional[dict]:
        """Get device configuration from DVR."""
        try:
            # Try to get device name and other config
            # For now, return basic info from device_info
            return {
                "model": "Hikvision DVR",
                "firmware": "Unknown"
            }
        except Exception as e:
            logger.error(f"Failed to get device config: {e}")
            return None
    
    def _discover_channels(
        self,
        login_id: int,
        device_info: NET_DVR_DEVICEINFO
    ) -> list[DiscoveredChannel]:
        """Discover all channels on the DVR."""
        channels = []
        start_channel = device_info.byStartChan
        total_channels = device_info.byChanNum
        
        logger.info(f"Device info: start_channel={start_channel}, total_channels={total_channels}")
        
        for i in range(total_channels):
            # Hikvision channels are typically 1-indexed
            channel_num = start_channel + i
            channel_name = self._get_channel_name(login_id, channel_num)
            status = self._get_channel_status(login_id, channel_num)
            
            channel = DiscoveredChannel(
                id=channel_num,
                name=channel_name or f"Camera {channel_num}",
                status=status,
                channel_type="analog",  # Simplified
                resolution=None
            )
            channels.append(channel)
        
        return channels
    
    def _get_channel_name(self, login_id: int, channel_num: int) -> str:
        """Get the name of a specific channel from DVR configuration."""
        try:
            from app.services.hcnetsdk_wrapper import _sdk_wrapper
            
            # Try to get channel name using NET_DVR_GetDVRConfig
            # This requires the channel configuration command
            # For now, return a more descriptive name
            return f"Channel {channel_num}"
        except Exception as e:
            logger.debug(f"Failed to get channel {channel_num} name: {e}")
            return f"Channel {channel_num}"
    
    def _get_channel_status(self, login_id: int, channel_num: int) -> str:
        """Get the status of a specific channel by attempting to start preview."""
        try:
            from app.services.hcnetsdk_wrapper import _sdk_wrapper
            
            # Try to start a brief preview to check if channel is online
            # Create a dummy callback
            def dummy_callback(l_real_handle: int, p_buffer, dw_buf_size: int, dw_user: int, p_reserved):
                pass
            
            from app.services.hcnetsdk_wrapper import REALDATA_CALLBACK
            callback = REALDATA_CALLBACK(dummy_callback)
            
            # Try to start preview
            success, play_handle = _sdk_wrapper.start_real_play(
                login_id,
                channel_num,
                callback,
                user_data=0
            )
            
            if success:
                # Stop the preview immediately
                _sdk_wrapper.stop_real_play(play_handle)
                return "online"
            else:
                error_code = _sdk_wrapper.get_last_error()
                logger.debug(f"Channel {channel_num} preview failed with error: {error_code}")
                # Error 107 means channel doesn't exist
                if error_code == 107:
                    return "offline"
                # For other errors (like error 7), mark as unknown so user can still try
                return "unknown"
        except Exception as e:
            logger.debug(f"Failed to get channel {channel_num} status: {e}")
            return "unknown"


# Singleton instance
_discovery_service = HikvisionDiscoveryService()


def discover_cameras(
    ip: str,
    port: int,
    username: str,
    password: str
) -> tuple[bool, Optional[DiscoveredDevice], str]:
    """
    Discover cameras on a Hikvision DVR.
    
    This is a convenience function that uses the singleton discovery service.
    """
    return _discovery_service.discover(ip, port, username, password)
