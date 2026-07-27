"""
hcnetsdk_wrapper.py
==================
Low-level wrapper for Hikvision HCNetSDK using ctypes.

This module provides Python bindings for the essential HCNetSDK functions
needed for DVR login, live preview, and stream decoding.
"""
from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Add DLL directory to search path
_dll_dir = os.path.join(os.path.dirname(__file__), "..", "..", "HCNetSDK")
_dll_dir = os.path.abspath(_dll_dir)
if os.path.exists(_dll_dir):
    os.add_dll_directory(_dll_dir)
    logger.info(f"Added DLL directory to search path: {_dll_dir}")
else:
    logger.warning(f"DLL directory not found: {_dll_dir}")


# ---------------------------------------------------------------------------
# HCNetSDK Structures
# ---------------------------------------------------------------------------
class NET_DVR_DEVICEINFO(ctypes.Structure):
    """Simpler device info structure for NET_DVR_Login (older API)."""
    _pack_ = 1
    _fields_ = [
        ("sSerialNumber", ctypes.c_char * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byMirrorChanNum", ctypes.c_byte),
        ("byStartChan1", ctypes.c_byte),
        ("byDecompressChanNum", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("bySupport3", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("bySupport5", ctypes.c_byte),
        ("bySupport6", ctypes.c_byte),
        ("bySupport7", ctypes.c_byte),
        ("bySupport8", ctypes.c_byte),
        ("wDevType", ctypes.c_ushort),
        ("bySupport9", ctypes.c_byte),
        ("byRes1", ctypes.c_byte * 16),
    ]


class NET_DVR_USER_LOGIN_INFO(ctypes.Structure):
    """Login information structure for NET_DVR_Login_V40 (per official Hikvision SDK)."""
    _pack_ = 1  # Disable padding for 64-bit compatibility
    _fields_ = [
        ("sDeviceAddress", ctypes.c_char * 128),
        ("byUseTransport", ctypes.c_byte),
        ("wPort", ctypes.c_ushort),
        ("sUserName", ctypes.c_char * 64),
        ("sPassword", ctypes.c_char * 64),
        ("cbLoginResult", ctypes.c_void_p),  # Callback function
        ("bUseAsynLogin", ctypes.c_byte),
        ("byLoginMode", ctypes.c_byte),
        ("byProxyType", ctypes.c_byte),
        ("sProxyParam", ctypes.c_char * 128),
        ("byRes", ctypes.c_byte * 256),
    ]


class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
    """Device information structure for NET_DVR_Login_V40."""
    _pack_ = 1  # Disable padding for 64-bit compatibility
    _fields_ = [
        ("sSerialNumber", ctypes.c_char * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byMirrorChanNum", ctypes.c_byte),
        ("byStartChan1", ctypes.c_byte),
        ("byDecompressChanNum", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("bySupport3", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("bySupport5", ctypes.c_byte),
        ("bySupport6", ctypes.c_byte),
        ("bySupport7", ctypes.c_byte),
        ("bySupport8", ctypes.c_byte),
        ("wDevType", ctypes.c_ushort),
        ("bySupport9", ctypes.c_byte),
        ("byRes1", ctypes.c_byte * 16),
    ]


class NET_DVR_PREVIEWINFO(ctypes.Structure):
    """Preview information structure for NET_DVR_RealPlay_V40."""
    _pack_ = 1  # Disable padding for 64-bit compatibility
    _fields_ = [
        ("hPlayWnd", ctypes.c_int),
        ("lChannel", ctypes.c_long),
        ("dwStreamType", ctypes.c_ulong),
        ("dwLinkMode", ctypes.c_ulong),
        ("byBlocked", ctypes.c_byte),
        ("byProtoType", ctypes.c_byte),
        ("sMultiCastIP", ctypes.c_char * 128),
        ("wPort", ctypes.c_ushort),
        ("byRes", ctypes.c_byte * 32),
    ]


# ---------------------------------------------------------------------------
# Callback Type (per official Hikvision SDK)
# ---------------------------------------------------------------------------
# fRealDataCallBack_V30: LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser
REALDATA_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_void_p,  # Return type
    ctypes.c_long,     # lRealHandle
    ctypes.c_ulong,    # dwDataType
    ctypes.POINTER(ctypes.c_byte),  # pBuffer
    ctypes.c_ulong,    # dwBufSize
    ctypes.c_void_p    # pUser
)


# ---------------------------------------------------------------------------
# HCNetSDK Wrapper Class
# ---------------------------------------------------------------------------
class HCNetSDKWrapper:
    """Wrapper for HCNetSDK.dll and PlayCtrl.dll."""
    
    def __init__(self):
        self.hcnetsdk = None
        self.playctrl = None
        self._initialized = False
        
    def load_dlls(self) -> bool:
        """Load HCNetSDK.dll and PlayCtrl.dll from common installation paths."""
        if self._initialized:
            return True
            
        try:
            # Try common installation paths for HCNetSDK
            sdk_paths = [
                r"C:\Program Files (x86)\Hikvision\Hikvision H.264 DVR\HCNetSDK.dll",
                r"C:\Program Files\Hikvision\Hikvision H.264 DVR\HCNetSDK.dll",
                r"C:\HCNetSDK\HCNetSDK.dll",
                r".\HCNetSDK.dll",
            ]
            
            playctrl_paths = [
                r"C:\Program Files (x86)\Hikvision\Hikvision H.264 DVR\PlayCtrl.dll",
                r"C:\Program Files\Hikvision\Hikvision H.264 DVR\PlayCtrl.dll",
                r"C:\HCNetSDK\PlayCtrl.dll",
                r".\PlayCtrl.dll",
            ]
            
            # Load HCNetSDK.dll
            self.hcnetsdk = None
            for path in sdk_paths:
                if os.path.exists(path):
                    try:
                        self.hcnetsdk = ctypes.cdll.LoadLibrary(path)
                        logger.info(f"Loaded HCNetSDK.dll from: {path}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load from {path}: {e}")
            
            if self.hcnetsdk is None:
                # Try loading from system PATH
                try:
                    self.hcnetsdk = ctypes.cdll.LoadLibrary("HCNetSDK.dll")
                    logger.info("Loaded HCNetSDK.dll from system PATH")
                except Exception as e:
                    logger.error(f"Failed to load HCNetSDK.dll: {e}")
                    return False
            
            # Load PlayCtrl.dll
            self.playctrl = None
            for path in playctrl_paths:
                if os.path.exists(path):
                    try:
                        self.playctrl = ctypes.cdll.LoadLibrary(path)
                        logger.info(f"Loaded PlayCtrl.dll from: {path}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load from {path}: {e}")
            
            if self.playctrl is None:
                try:
                    self.playctrl = ctypes.cdll.LoadLibrary("PlayCtrl.dll")
                    logger.info("Loaded PlayCtrl.dll from system PATH")
                except Exception as e:
                    logger.error(f"Failed to load PlayCtrl.dll: {e}")
                    return False
            
            # Configure function signatures
            self._configure_functions()
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to load HCNetSDK DLLs: {e}")
            return False
    
    def _configure_functions(self):
        """Configure function signatures for HCNetSDK functions."""
        # NET_DVR_Init
        self.hcnetsdk.NET_DVR_Init.restype = ctypes.c_bool
        self.hcnetsdk.NET_DVR_Init.argtypes = []
        
        # NET_DVR_Cleanup
        self.hcnetsdk.NET_DVR_Cleanup.restype = ctypes.c_bool
        self.hcnetsdk.NET_DVR_Cleanup.argtypes = []
        
        # NET_DVR_GetLastError
        self.hcnetsdk.NET_DVR_GetLastError.restype = ctypes.c_long
        self.hcnetsdk.NET_DVR_GetLastError.argtypes = []
        
        # NET_DVR_Login (simpler, older API - less prone to alignment issues)
        self.hcnetsdk.NET_DVR_Login.restype = ctypes.c_long
        self.hcnetsdk.NET_DVR_Login.argtypes = [
            ctypes.c_char_p,  # sDVRIP
            ctypes.c_ushort,  # wDVRPort
            ctypes.c_char_p,  # sUserName
            ctypes.c_char_p,  # sPassword
            ctypes.POINTER(NET_DVR_DEVICEINFO)  # lpDeviceInfo
        ]
        
        # NET_DVR_Login_V40 (per official Hikvision SDK)
        self.hcnetsdk.NET_DVR_Login_V40.restype = ctypes.c_long
        self.hcnetsdk.NET_DVR_Login_V40.argtypes = [
            ctypes.POINTER(NET_DVR_USER_LOGIN_INFO),  # pLoginInfo
            ctypes.POINTER(NET_DVR_DEVICEINFO_V40)  # lpDeviceInfo
        ]
        
        # NET_DVR_Logout
        self.hcnetsdk.NET_DVR_Logout.restype = ctypes.c_bool
        self.hcnetsdk.NET_DVR_Logout.argtypes = [ctypes.c_long]
        
        # NET_DVR_RealPlay_V40
        self.hcnetsdk.NET_DVR_RealPlay_V40.restype = ctypes.c_long
        self.hcnetsdk.NET_DVR_RealPlay_V40.argtypes = [
            ctypes.c_long,  # lUserID
            ctypes.POINTER(NET_DVR_PREVIEWINFO),  # lpPreviewInfo
            REALDATA_CALLBACK,  # fRealDataCallBack_V30
            ctypes.c_void_p  # pUser
        ]
        
        # NET_DVR_StopRealPlay
        self.hcnetsdk.NET_DVR_StopRealPlay.restype = ctypes.c_bool
        self.hcnetsdk.NET_DVR_StopRealPlay.argtypes = [ctypes.c_long]
        
        # NET_DVR_SetRealDataCallBack
        self.hcnetsdk.NET_DVR_SetRealDataCallBack.restype = ctypes.c_bool
        self.hcnetsdk.NET_DVR_SetRealDataCallBack.argtypes = [
            ctypes.c_long,  # lRealHandle
            REALDATA_CALLBACK,  # fRealDataCallBack_V30
            ctypes.c_ulong  # dwUser
        ]
        
        # PlayCtrl functions
        if self.playctrl:
            self.playctrl.PlayM4_GetPort.restype = ctypes.c_long
            self.playctrl.PlayM4_GetPort.argtypes = []
            
            self.playctrl.PlayM4_FreePort.restype = ctypes.c_bool
            self.playctrl.PlayM4_FreePort.argtypes = [ctypes.c_long]
            
            self.playctrl.PlayM4_OpenStream.restype = ctypes.c_bool
            self.playctrl.PlayM4_OpenStream.argtypes = [
                ctypes.c_long,  # nPort
                ctypes.c_void_p,  # pFileHead
                ctypes.c_ulong,  # nSize
                ctypes.c_ulong  # nBufPoolSize
            ]
            
            self.playctrl.PlayM4_SetDisplayBuf.restype = ctypes.c_bool
            self.playctrl.PlayM4_SetDisplayBuf.argtypes = [ctypes.c_long, ctypes.c_int]
            
            self.playctrl.PlayM4_Play.restype = ctypes.c_bool
            self.playctrl.PlayM4_Play.argtypes = [ctypes.c_long, ctypes.c_void_p]
            
            self.playctrl.PlayM4_Stop.restype = ctypes.c_bool
            self.playctrl.PlayM4_Stop.argtypes = [ctypes.c_long]
            
            self.playctrl.PlayM4_InputData.restype = ctypes.c_bool
            self.playctrl.PlayM4_InputData.argtypes = [
                ctypes.c_long,  # nPort
                ctypes.POINTER(ctypes.c_byte),  # pBuf
                ctypes.c_ulong  # nSize
            ]
            
            self.playctrl.PlayM4_CloseStream.restype = ctypes.c_bool
            self.playctrl.PlayM4_CloseStream.argtypes = [ctypes.c_long]
            
            # PlayM4_GetPicture may not be available in all SDK versions - make it optional
            try:
                self.playctrl.PlayM4_GetPicture.restype = ctypes.c_bool
                self.playctrl.PlayM4_GetPicture.argtypes = [
                    ctypes.c_long,  # nPort
                    ctypes.POINTER(ctypes.c_byte),  # pBuf
                    ctypes.POINTER(ctypes.c_ulong)  # pSize
                ]
                self._has_get_picture = True
            except AttributeError:
                logger.warning("PlayM4_GetPicture not available in this PlayCtrl version")
                self._has_get_picture = False
    
    def initialize(self) -> bool:
        """Initialize HCNetSDK."""
        if not self.load_dlls():
            return False
            
        try:
            if not self.hcnetsdk.NET_DVR_Init():
                error_code = self.hcnetsdk.NET_DVR_GetLastError()
                logger.error(f"NET_DVR_Init failed with error code: {error_code}")
                return False
            
            self._initialized = True
            logger.info("HCNetSDK initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize HCNetSDK: {e}")
            return False
    
    def cleanup(self):
        """Cleanup HCNetSDK resources."""
        if self._initialized and self.hcnetsdk:
            try:
                self.hcnetsdk.NET_DVR_Cleanup()
                logger.info("HCNetSDK cleanup completed")
            except Exception as e:
                logger.error(f"Error during HCNetSDK cleanup: {e}")
        self._initialized = False
    
    def get_last_error(self) -> int:
        """Get last SDK error code."""
        if self.hcnetsdk:
            return self.hcnetsdk.NET_DVR_GetLastError()
        return -1
    
    def login(
        self,
        dvr_ip: str,
        dvr_port: int,
        username: str,
        password: str
    ) -> tuple[bool, int, Optional[NET_DVR_DEVICEINFO]]:
        """Login to DVR using simpler NET_DVR_Login API. Returns (success, login_id, device_info)."""
        if not self._initialized:
            return False, -1, None
            
        try:
            device_info = NET_DVR_DEVICEINFO()
            ip_bytes = dvr_ip.encode('utf-8')
            user_bytes = username.encode('utf-8')
            pass_bytes = password.encode('utf-8')
            
            # Try NET_DVR_Login_V40 first (better authentication support)
            try:
                device_info_v40 = NET_DVR_DEVICEINFO_V40()
                login_info = NET_DVR_USER_LOGIN_INFO()
                # Initialize all fields to zero
                ctypes.memset(ctypes.byref(login_info), 0, ctypes.sizeof(login_info))
                # Set required fields per official SDK
                ctypes.memmove(ctypes.byref(login_info.sDeviceAddress), ip_bytes, min(len(ip_bytes), 128))
                login_info.wPort = dvr_port
                ctypes.memmove(ctypes.byref(login_info.sUserName), user_bytes, min(len(user_bytes), 64))
                ctypes.memmove(ctypes.byref(login_info.sPassword), pass_bytes, min(len(pass_bytes), 64))
                login_info.bUseAsynLogin = 0  # Synchronous login
                login_info.byLoginMode = 0  # Default login mode
                
                login_id = self.hcnetsdk.NET_DVR_Login_V40(
                    ctypes.byref(login_info),
                    ctypes.byref(device_info_v40)
                )
                
                if login_id > 0:  # LoginID must be positive, 0 is invalid
                    logger.info(
                        f"Successfully logged in via V40 to {dvr_ip}:{dvr_port} "
                        f"(device type: {device_info_v40.wDevType}, channels: {device_info_v40.byChanNum})"
                    )
                    # Convert V40 info to simple device info for compatibility
                    device_info.byChanNum = device_info_v40.byChanNum
                    device_info.byStartChan = device_info_v40.byStartChan
                    device_info.wDevType = device_info_v40.wDevType
                    return True, login_id, device_info
            except Exception as e:
                logger.debug(f"NET_DVR_Login_V40 failed, trying NET_DVR_Login: {e}")
            
            # Fallback to NET_DVR_Login
            login_id = self.hcnetsdk.NET_DVR_Login(
                ip_bytes,
                dvr_port,
                user_bytes,
                pass_bytes,
                ctypes.byref(device_info)
            )
            
            if login_id <= 0:  # LoginID must be positive, 0 or negative is invalid
                error_code = self.get_last_error()
                if login_id == 0:
                    logger.error(f"NET_DVR_Login returned LoginID=0 (invalid), treating as failure")
                # Common HCNetSDK error codes
                error_messages = {
                    1: "Password error",
                    17: "Password error or account disabled",
                    7: "No privilege - User account lacks admin permissions. Please use the admin account.",
                    8: "Invalid parameter",
                    91: "Sub-system not supported",
                    92: "Only support sub-system",
                }
                error_msg = error_messages.get(error_code, f"Unknown error code: {error_code}")
                logger.error(f"Login failed for {dvr_ip}:{dvr_port} - {error_msg}")
                return False, -1, None
            
            logger.info(
                f"Successfully logged in to {dvr_ip}:{dvr_port} "
                f"(device type: {device_info.wDevType}, channels: {device_info.byChanNum})"
            )
            return True, login_id, device_info
            
        except Exception as e:
            logger.exception(f"Login exception for {dvr_ip}:{dvr_port}: {e}")
            return False, -1, None
    
    def logout(self, login_id: int) -> bool:
        """Logout from DVR."""
        if not self._initialized or login_id < 0:
            return False
            
        try:
            result = self.hcnetsdk.NET_DVR_Logout(login_id)
            if result:
                logger.debug(f"Successfully logged out (login_id: {login_id})")
            else:
                logger.warning(f"Logout failed (login_id: {login_id})")
            return result
        except Exception as e:
            logger.exception(f"Logout exception: {e}")
            return False
    
    def start_real_play(
        self,
        login_id: int,
        channel: int,
        callback: REALDATA_CALLBACK,
        user_data: int = 0,
        start_channel: int = 1
    ) -> tuple[bool, int]:
        """Start real-time preview. Returns (success, play_handle)."""
        if not self._initialized or login_id < 0:
            return False, -1
            
        try:
            preview_info = NET_DVR_PREVIEWINFO()
            # Initialize all fields to zero
            ctypes.memset(ctypes.byref(preview_info), 0, ctypes.sizeof(preview_info))
            preview_info.hPlayWnd = 0  # No window, we use callback
            # Channel numbering: for analog DVRs, lChannel = start_channel + (channel - 1)
            # This matches official Hikvision SDK behavior
            preview_info.lChannel = start_channel + (channel - 1)
            preview_info.dwStreamType = 0  # Main stream (0=Main, 1=Sub, 2=Third)
            preview_info.dwLinkMode = 0  # TCP (0=TCP, 1=UDP, 2=Multicast, 3=RTP, 4=RTP/RTSP, 5=RTSP/HTTP)
            preview_info.byBlocked = 0  # Non-blocking mode (0=Non-blocking, 1=Blocking)
            preview_info.byProtoType = 0  # Private protocol (0=Private, 1=RTSP, 2=Auto)
            
            logger.info(
                f"Starting preview - LoginID: {login_id}, RequestedChannel: {channel}, "
                f"SDKChannel: {preview_info.lChannel}, StartChannel: {start_channel}, "
                f"StreamType: Main, LinkMode: TCP, ProtoType: Private, Blocked: False"
            )
            
            # Pass callback directly to NET_DVR_RealPlay_V40 as per official SDK
            play_handle = self.hcnetsdk.NET_DVR_RealPlay_V40(
                login_id,
                ctypes.byref(preview_info),
                callback,
                ctypes.c_void_p(user_data)
            )
            
            if play_handle < 0:
                error_code = self.get_last_error()
                error_messages = {
                    1: "Password error",
                    7: "No privilege",
                    17: "Password error or account disabled",
                    91: "Sub-system not supported",
                    92: "Only support sub-system",
                    107: "Channel error (channel doesn't exist or invalid)",
                    108: "Preview failed",
                }
                error_msg = error_messages.get(error_code, f"Unknown error code: {error_code}")
                logger.error(
                    f"NET_DVR_RealPlay_V40 failed - Error Code: {error_code}, "
                    f"Meaning: {error_msg}, Channel: {channel}, LoginID: {login_id}"
                )
                return False, -1
            
            logger.info(
                f"Successfully started real play - Channel: {channel}, "
                f"PlayHandle: {play_handle}, LoginID: {login_id}"
            )
            return True, play_handle
            
        except Exception as e:
            logger.exception(f"Start real play exception: {e}")
            return False, -1
    
    def stop_real_play(self, play_handle: int) -> bool:
        """Stop real-time preview."""
        if not self._initialized or play_handle < 0:
            return False
            
        try:
            result = self.hcnetsdk.NET_DVR_StopRealPlay(play_handle)
            if result:
                logger.debug(f"Stopped real play (play_handle: {play_handle})")
            else:
                logger.warning(f"Stop real play failed (play_handle: {play_handle})")
            return result
        except Exception as e:
            logger.exception(f"Stop real play exception: {e}")
            return False


# Global SDK wrapper instance
_sdk_wrapper = HCNetSDKWrapper()
