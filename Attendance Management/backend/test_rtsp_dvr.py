"""
Standalone RTSP DVR Tester for Attendance Management System
Tests connection to Hikvision DVR and displays live video stream
"""

import cv2
import time
import sys
from datetime import datetime

# Hikvision DVR RTSP URL formats to test
HIKVISION_URL_FORMATS = [
    # Standard Hikvision format (Channel 1-4)
    "rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel}01",
    "rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel}02",  # Sub-stream
    
    # Alternative Hikvision formats
    "rtsp://{username}:{password}@{ip}:554/h264/ch{channel}/main/av_stream",
    "rtsp://{username}:{password}@{ip}:554/h264/ch{channel}/sub/av_stream",
    
    # Older Hikvision DVR formats
    "rtsp://{username}:{password}@{ip}:554/PSIP/stream{channel}",
    "rtsp://{username}:{password}@{ip}:554/MPEG-4/ch{channel}/main/av_stream",
    
    # Generic formats
    "rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0",
    "rtsp://{username}:{password}@{ip}:554/live",
    "rtsp://{username}:{password}@{ip}:554/stream{channel}",
]

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_rtsp_url(url, timeout=10, display=False):
    """
    Test a single RTSP URL with detailed logging
    
    Args:
        url: RTSP URL to test
        timeout: Connection timeout in seconds
        display: Whether to display video window
    
    Returns:
        dict: Test results with status and details
    """
    log(f"Testing URL: {url}")
    log(f"OpenCV Version: {cv2.__version__}")
    
    # Check FFmpeg support
    build_info = cv2.getBuildInformation()
    ffmpeg_supported = "FFmpeg" in build_info
    log(f"FFmpeg Support: {ffmpeg_supported}")
    
    if not ffmpeg_supported:
        log("ERROR: OpenCV built without FFmpeg support - RTSP will not work")
        return {"success": False, "error": "OpenCV lacks FFmpeg support"}
    
    cap = None
    try:
        # Try opening with FFmpeg backend explicitly
        log("Attempting to open stream with FFmpeg backend...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set timeout (works on some backends)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
        
        if not cap.isOpened():
            log("ERROR: Could not open VideoCapture")
            return {"success": False, "error": "Could not open stream"}
        
        log("✓ VideoCapture opened successfully")
        
        # Read first frame with timeout
        log("Attempting to read first frame...")
        start_time = time.time()
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        
        if not ret or frame is None:
            log(f"ERROR: Could not read frame after {elapsed:.2f}s")
            return {"success": False, "error": "Could not read frame"}
        
        log(f"✓ First frame read successfully in {elapsed:.2f}s")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = cap.get(cv2.CAP_PROP_FOURCC)
        
        log(f"Stream properties:")
        log(f"  Resolution: {width}x{height}")
        log(f"  FPS: {fps}")
        log(f"  Codec: {codec}")
        
        result = {
            "success": True,
            "width": width,
            "height": height,
            "fps": fps,
            "codec": codec,
            "first_frame_time": elapsed
        }
        
        # Display video if requested
        if display:
            log("Starting video display (press 'q' to quit)...")
            frame_count = 0
            start_display = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    log("ERROR: Lost connection during display")
                    break
                
                frame_count += 1
                cv2.imshow(f"DVR Stream - {url}", frame)
                
                # Display stats
                if frame_count % 30 == 0:
                    elapsed_display = time.time() - start_display
                    actual_fps = frame_count / elapsed_display
                    log(f"Display stats: {frame_count} frames, {actual_fps:.1f} FPS")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log("User requested quit")
                    break
            
            cv2.destroyAllWindows()
            log(f"Display stopped. Total frames: {frame_count}")
        
        return result
        
    except Exception as e:
        log(f"EXCEPTION: {type(e).__name__}: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        if cap is not None:
            cap.release()
            log("VideoCapture released")

def test_hikvision_dvr(ip, username, password, channels=[1, 2, 3, 4], timeout=10):
    """
    Test all common Hikvision DVR RTSP URL formats
    
    Args:
        ip: DVR IP address
        username: DVR username
        password: DVR password
        channels: List of channel numbers to test
        timeout: Connection timeout in seconds
    """
    log("=" * 70)
    log("HIKVISION DVR RTSP CONNECTION TESTER - ATTENDANCE SYSTEM")
    log("=" * 70)
    log(f"DVR IP: {ip}")
    log(f"Username: {username}")
    log(f"Password: {'*' * len(password)}")
    log(f"Channels to test: {channels}")
    log(f"Timeout: {timeout}s")
    log("=" * 70)
    
    successful_urls = []
    
    for channel in channels:
        log(f"\n--- Testing Channel {channel} ---")
        
        for url_template in HIKVISION_URL_FORMATS:
            url = url_template.format(
                username=username,
                password=password,
                ip=ip,
                channel=channel
            )
            
            result = test_rtsp_url(url, timeout=timeout, display=False)
            
            if result["success"]:
                log(f"✓✓✓ SUCCESS: {url}")
                log(f"  Resolution: {result['width']}x{result['height']}")
                log(f"  FPS: {result['fps']}")
                successful_urls.append(url)
                
                # Ask if user wants to see live video
                try:
                    response = input(f"\nDisplay live video for this stream? (y/n): ").strip().lower()
                    if response == 'y':
                        test_rtsp_url(url, timeout=timeout, display=True)
                except (EOFError, KeyboardInterrupt):
                    log("Skipping display")
                
                # Skip other formats for this channel if one works
                break
            else:
                log(f"✗ Failed: {result['error']}")
    
    log("\n" + "=" * 70)
    log("TEST SUMMARY")
    log("=" * 70)
    
    if successful_urls:
        log(f"✓ Found {len(successful_urls)} working URL(s):")
        for url in successful_urls:
            log(f"  - {url}")
        
        log("\n" + "=" * 70)
        log("CONFIGURATION FOR ATTENDANCE SYSTEM")
        log("=" * 70)
        log("Use these URLs in the CCTV Attendance page:")
        log("1. Navigate to CCTV Attendance page")
        log("2. Enter the working RTSP URL in the 'CCTV stream URL' field")
        log("3. Set Camera ID (e.g., 'gate-1', 'entrance-2')")
        log("4. Select Camera Purpose (Check-in or Check-out)")
        log("5. Click 'Test connection' to verify")
        log("6. Click 'Start auto-scan' to begin attendance marking")
    else:
        log("✗ No working URLs found")
        log("\nTroubleshooting suggestions:")
        log("1. Verify DVR IP address is correct")
        log("2. Check username and password")
        log("3. Ensure DVR has RTSP enabled in settings")
        log("4. Check network connectivity (ping the DVR)")
        log("5. Verify DVR firmware is up to date")
        log("6. Check if DVR uses non-standard RTSP port")
        log("7. Try different channel numbers (1-4 or 101-104)")
    
    return successful_urls

if __name__ == "__main__":
    print("Hikvision DVR RTSP Tester - Attendance Management System")
    print("=" * 70)
    
    try:
        ip = input("DVR IP Address [192.168.29.181]: ").strip() or "192.168.29.181"
        username = input("Username [anilchanna@gmail.com]: ").strip() or "anilchanna@gmail.com"
        password = input("Password [8eGd3P2o26]: ").strip() or "8eGd3P2o26"
        
        channels_input = input("Channels to test (comma-separated, e.g., 1,2,3,4) [1]: ").strip()
        if channels_input:
            channels = [int(c.strip()) for c in channels_input.split(",")]
        else:
            channels = [1]
        
        timeout_input = input("Timeout seconds [10]: ").strip()
        timeout = int(timeout_input) if timeout_input else 10
        
        test_hikvision_dvr(ip, username, password, channels, timeout)
        
    except KeyboardInterrupt:
        log("\nTest interrupted by user")
    except Exception as e:
        log(f"Error: {e}")
