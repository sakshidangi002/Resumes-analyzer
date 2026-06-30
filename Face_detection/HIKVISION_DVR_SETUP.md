# Hikvision DVR RTSP Configuration Guide

## Overview
This guide explains how to configure Hikvision DVRs for RTSP streaming with the Face Attendance System.

## Your Setup
- **Camera Model:** Hikvision DS-2CE76D0T-LPFS (Analog Camera)
- **DVR IP:** 192.168.29.181
- **Connection:** 4 analog cameras → Hikvision DVR → Network
- **Protocol:** RTSP (Real Time Streaming Protocol)

## Important Notes
- Analog cameras DO NOT have individual IP addresses
- All cameras are accessed through the DVR using RTSP channels
- The DVR acts as the video encoder and streaming server

## Enabling RTSP on Hikvision DVR

### Step 1: Access DVR Web Interface
1. Open browser and navigate to: `http://192.168.29.181`
2. Login with admin credentials
3. Navigate to: **Configuration** → **Network** → **RTSP**

### Step 2: Configure RTSP Settings
- **Enable RTSP:** Check/Enable
- **RTSP Port:** Default is 554 (can be changed if needed)
- **Max Connections:** Set to at least 4 (one per camera)
- **Authentication:** Basic or Digest (Basic is simpler for OpenCV)

### Step 3: Set Stream Parameters
Navigate to: **Configuration** → **Video & Audio** → **Video**

For each channel (1-4):
- **Video Type:** H.264 (recommended for OpenCV compatibility)
- **Resolution:** 1080P or 720P (higher resolution = better face recognition)
- **Frame Rate:** 15-25 FPS (sufficient for face recognition)
- **Bitrate:** 2048-4096 Kbps (balance quality vs bandwidth)

### Step 4: Save and Reboot
- Click **Apply** or **Save**
- Reboot DVR if required

## RTSP URL Formats for Hikvision DVRs

### Standard Format (Most Common)
```
rtsp://username:password@192.168.29.181:554/Streaming/Channels/101
```

- `101` = Channel 1, Main stream
- `102` = Channel 1, Sub-stream (lower resolution)
- `201` = Channel 2, Main stream
- `202` = Channel 2, Sub-stream
- etc.

### Alternative Hikvision Formats
```
# Format 2: Alternative channel notation
rtsp://username:password@192.168.29.181:554/h264/ch1/main/av_stream
rtsp://username:password@192.168.29.181:554/h264/ch1/sub/av_stream

# Format 3: RealMonitor format
rtsp://username:password@192.168.29.181:554/cam/realmonitor?channel=1&subtype=0

# Format 4: MPEG-4 format (older DVRs)
rtsp://username:password@192.168.29.181:554/MPEG-4/ch1/main/av_stream

# Format 5: PSIP format (older DVRs)
rtsp://username:password@192.168.29.181:554/PSIP/stream1
```

### Your Specific URLs
Based on your credentials, test these URLs in order:

**Channel 1 (Check-in Camera):**
```
rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/Streaming/Channels/101
rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/h264/ch1/main/av_stream
rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/cam/realmonitor?channel=1&subtype=0
```

**Channel 2 (Check-out Camera):**
```
rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/Streaming/Channels/201
rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/h264/ch2/main/av_stream
rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/cam/realmonitor?channel=2&subtype=0
```

## Channel Mapping for Attendance System

### Recommended Configuration
| Camera ID | Camera Name | DVR Channel | RTSP URL | Purpose |
|-----------|-------------|-------------|----------|---------|
| 1 | Main Entrance Check-in | Channel 1 | `/Streaming/Channels/101` | Check-in |
| 2 | Main Exit Check-out | Channel 2 | `/Streaming/Channels/201` | Check-out |
| 3 | Side Door Check-in | Channel 3 | `/Streaming/Channels/301` | Check-in |
| 4 | Parking Area | Channel 4 | `/Streaming/Channels/401` | Check-in |

## Testing RTSP Connection

### Using the Provided Test Script
```bash
cd Face_detection
python test_rtsp_dvr.py
```

The script will:
- Test all common Hikvision URL formats automatically
- Display detailed connection information
- Show live video preview if connection succeeds
- Report stream properties (resolution, FPS, codec)

### Manual Testing with VLC Player
1. Open VLC Media Player
2. Go to: **Media** → **Open Network Stream**
3. Paste RTSP URL
4. Click **Play**

If VLC can play the stream, OpenCV should also work.

### Manual Testing with FFmpeg
```bash
ffplay rtsp://anilchanna@gmail.com:8eGd3P2o26@192.168.29.181:554/Streaming/Channels/101
```

## Troubleshooting RTSP Issues

### Error: 401 Unauthorized
- **Cause:** Incorrect username or password
- **Solution:** Verify DVR credentials in web interface
- **Note:** Some DVRs use different credentials for RTSP vs web interface

### Error: 404 Not Found
- **Cause:** Incorrect RTSP path or channel number
- **Solution:** Try different URL formats listed above
- **Check:** Verify channel numbers in DVR web interface

### Error: Connection Timeout
- **Cause:** Network connectivity issue or DVR not responding
- **Solution:**
  - Ping the DVR: `ping 192.168.29.181`
  - Check firewall settings
  - Verify DVR is powered on and connected

### Error: Could Not Read Frame
- **Cause:** Stream opened but no video data
- **Solution:**
  - Check if camera is connected to DVR
  - Verify video encoding is H.264 (not H.265)
  - Check bitrate settings

### Error: Unsupported Codec
- **Cause:** DVR using H.265 codec not supported by OpenCV
- **Solution:** Change DVR encoding to H.264 in video settings

## Network Configuration

### Firewall Settings
Ensure the following ports are open:
- **TCP 554** (RTSP)
- **TCP 80** (HTTP for DVR web interface)
- **TCP 8000** (Alternative RTSP port on some DVRs)

### Bandwidth Requirements
- **1080P @ 15 FPS:** ~2-4 Mbps per camera
- **720P @ 15 FPS:** ~1-2 Mbps per camera
- **Total for 4 cameras:** 8-16 Mbps

### Network Topology
```
Analog Cameras (BNC cables)
    ↓
Hikvision DVR (192.168.29.181)
    ↓
Network Switch
    ↓
Face Attendance Server
```

## DVR-Specific Settings

### Hikvision DVR/iDS-7200 Series
- **RTSP Path:** `/Streaming/Channels/`
- **Default Port:** 554
- **Authentication:** Basic or Digest

### Hikvision NVR (Network Video Recorder)
- **RTSP Path:** `/Streaming/Channels/` or `/h264/`
- **Default Port:** 554
- **Authentication:** Basic

### Older Hikvision DVR Models
- **RTSP Path:** `/PSIP/stream` or `/MPEG-4/ch`
- **Default Port:** 554
- **Authentication:** Basic

## Best Practices

1. **Use Main Stream for Recognition:** Higher resolution = better accuracy
2. **Set Appropriate FPS:** 15-25 FPS is sufficient for face recognition
3. **Use Sub-stream for Preview:** Lower bandwidth for monitoring
4. **Enable Authentication:** Prevent unauthorized access
5. **Monitor Bandwidth:** Ensure network can handle multiple streams
6. **Test Each Channel:** Verify all cameras work before deployment
7. **Document Working URLs:** Save successful RTSP URLs for reference

## Security Considerations

1. **Change Default Passwords:** Use strong, unique passwords
2. **Enable HTTPS:** Use HTTPS for DVR web interface if available
3. **Network Isolation:** Place DVRs on separate VLAN if possible
4. **Regular Updates:** Keep DVR firmware updated
5. **Access Control:** Limit IP addresses that can access RTSP streams

## Integration with Face Attendance System

Once RTSP is working:

1. **Add Cameras in Web Interface:**
   - Navigate to CCTV Cameras page
   - Click "Add Camera"
   - Enter camera name (e.g., "Main Entrance Check-in")
   - Select camera purpose (IN or OUT)
   - Paste working RTSP URL
   - Click "Test connection"
   - Click "Add and start camera"

2. **Verify Recognition:**
   - Check camera preview in web interface
   - Verify face detection boxes appear
   - Test with registered employees

3. **Monitor Performance:**
   - Check camera status in dashboard
   - Monitor frame counts and reconnect stats
   - Review error logs if issues occur

## Additional Resources

- Hikvision RTSP Documentation: [Hikvision Official Docs]
- OpenCV RTSP Support: [OpenCV Documentation]
- FFmpeg RTSP Options: [FFmpeg Documentation]
