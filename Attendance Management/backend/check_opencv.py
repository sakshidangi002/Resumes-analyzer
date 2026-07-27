import cv2
import sys

print("=" * 60)
print("OpenCV Build Information Check")
print("=" * 60)
print(f"OpenCV Version: {cv2.__version__}")
print()

build_info = cv2.getBuildInformation()

# Check for FFmpeg support
ffmpeg_support = "FFmpeg" in build_info
print(f"FFmpeg Support: {ffmpeg_support}")

# Check for RTSP support
rtsp_support = "RTSP" in build_info or "rtsp" in build_info.lower()
print(f"RTSP Support: {rtsp_support}")

# Check for H264/H265 support
h264_support = "H264" in build_info or "h264" in build_info.lower()
h265_support = "H265" in build_info or "h265" in build_info.lower()
print(f"H264 Support: {h264_support}")
print(f"H265 Support: {h265_support}")

# Check backend availability
print()
print("Available Video Backends:")
backends = {
    cv2.CAP_FFMPEG: "FFMPEG",
    cv2.CAP_GSTREAMER: "GSTREAMER",
    cv2.CAP_DSHOW: "DirectShow",
    cv2.CAP_MSMF: "Media Foundation",
    cv2.CAP_V4L: "V4L2",
}

for backend_id, backend_name in backends.items():
    try:
        # Try to create a capture with this backend
        cap = cv2.VideoCapture(0, backend_id)
        if cap.isOpened():
            print(f"  ✓ {backend_name} (ID: {backend_id})")
            cap.release()
        else:
            print(f"  ✗ {backend_name} (ID: {backend_id}) - Not available")
    except:
        print(f"  ✗ {backend_name} (ID: {backend_id}) - Error checking")

print()
print("=" * 60)
