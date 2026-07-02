# HCNetSDK Integration Setup Guide

This guide explains how to set up Hikvision HCNetSDK for direct DVR connection in the Attendance Management system.

## Overview

The HCNetSDK integration replaces RTSP/OpenCV video capture with Hikvision's native SDK for more reliable streaming from Hikvision DVRs. This is particularly useful when RTSP streams have decoding compatibility issues.

## Prerequisites

### Hardware

- Hikvision DVR (tested with DS-7A08HGHI-F1/ECO)
- Hikvision analog or IP cameras connected to the DVR
- Windows operating system (HCNetSDK is Windows-only)

### Software

- Python 3.8+
- Attendance Management backend
- Hikvision HCNetSDK library

## Installation

### Step 1: Download HCNetSDK

1. Visit the Hikvision official website: https://www.hikvision.com/en/support/download/
2. Search for "HCNetSDK" or navigate to: Software > Application Tools > HCNetSDK
3. Download the latest version of HCNetSDK for Windows
4. Extract the downloaded archive

### Step 2: Install HCNetSDK DLLs

The application will automatically search for HCNetSDK.dll and PlayCtrl.dll in the following locations (in order):

1. `C:\Program Files (x86)\Hikvision\Hikvision H.264 DVR\HCNetSDK.dll`
2. `C:\Program Files\Hikvision\Hikvision H.264 DVR\HCNetSDK.dll`
3. `C:\HCNetSDK\HCNetSDK.dll`
4. `.\HCNetSDK.dll` (application directory)
5. System PATH

**Recommended installation:**

Copy the following files from the extracted HCNetSDK package to one of the above locations:

- `HCNetSDK.dll`
- `PlayCtrl.dll`
- `PlayM4.dll` (if included)
- `HCCore.dll` (if included)

**Simplest approach:**
```bash
# Create a folder in your application directory
mkdir HCNetSDK

# Copy DLLs from extracted SDK to this folder
# From: C:\path\to\extracted\HCNetSDK\library
# To: C:\sakshi folder\application\Resume analyzer\Attendance Management\backend\HCNetSDK
```

### Step 3: Verify Installation

The application will log whether HCNetSDK was successfully loaded on startup. Check the logs for:

```
INFO: Loaded HCNetSDK.dll from: C:\HCNetSDK\HCNetSDK.dll
INFO: Loaded PlayCtrl.dll from: C:\HCNetSDK\PlayCtrl.dll
INFO: HCNetSDK initialized successfully
```

If you see warnings like:
```
WARNING: HCNetSDK camera module not available
```

Then the DLLs were not found. Check the file paths and ensure they exist.

## Configuration

### Database Configuration

To use HCNetSDK for a camera, update the camera configuration in the database:

```sql
UPDATE cameras 
SET source_type = 'hcnetsdk',
    source_url = 'hcnetsdk://192.168.1.100:8000@admin:password123?channel=1'
WHERE id = 1;
```

### URL Format

The `source_url` for HCNetSDK cameras uses this format:

```
hcnetsdk://<dvr_ip>:<dvr_port>@<username>:<password>?channel=<channel_number>
```

**Example:**
```
hcnetsdk://192.168.1.100:8000@admin:password123?channel=1
```

**Parameters:**
- `dvr_ip`: DVR IP address (e.g., 192.168.1.100)
- `dvr_port`: DVR server port (default: 8000)
- `username`: DVR username
- `password`: DVR password
- `channel`: Camera channel number (1-16 depending on DVR)

### API Configuration

You can also configure cameras via the REST API:

```bash
curl -X PUT http://localhost:8000/api/cameras/1 \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "hcnetsdk",
    "source_url": "hcnetsdk://192.168.1.100:8000@admin:password123?channel=1",
    "enabled": true
  }'
```

## DVR Configuration

### Enable RTSP/Streaming

1. Log in to your DVR web interface (usually http://<dvr_ip>)
2. Navigate to: Configuration > Network > Basic Settings
3. Ensure the server port is set to 8000 (or your desired port)
4. Enable TCP transport for better reliability

### User Permissions

Ensure the DVR user account has:
- View live video permissions
- Access to the specified camera channel
- Local network access

## Troubleshooting

### DLL NotFound Error

**Symptom:** `WARNING: HCNetSDK camera module not available`

**Solution:**
1. Verify DLL files exist in one of the search paths
2. Check file permissions
3. Try copying DLLs to the application directory
4. Add DLL directory to Windows PATH environment variable

### Login Failed

**Symptom:** `Login failed (error code: <code>)`

**Common error codes:**
- `1`: Invalid username or password
- `2`: User has no permission
- `3`: Connection failed
- `4`: Device not found

**Solution:**
1. Verify DVR IP and port are correct
2. Check DVR is accessible from the application server
3. Verify username and password
4. Check user permissions on DVR

### Preview Failed

**Symptom:** `Start real play failed (error code: <code>)`

**Solution:**
1. Verify channel number is valid (1 to max channels)
2. Check if channel is enabled in DVR
3. Ensure camera is connected to that channel
4. Try different stream type (main vs sub stream)

### No Frames Received

**Symptom:** Camera connects but no frames are decoded

**Solution:**
1. Check PlayCtrl.dll is loaded correctly
2. Verify H.264 codec is supported
3. Check DVR stream format settings
4. Try different channel or stream type
5. Check network connectivity

### Green/Purple Blocks (if falling back to RTSP)

**Symptom:** Corrupted video frames with green/purple artifacts

**Solution:**
This is the exact issue HCNetSDK solves. Ensure:
1. `source_type` is set to `hcnetsdk` (not `rtsp`)
2. HCNetSDK DLLs are loaded successfully
3. Check logs confirm HCNetSDK is being used

## Logging

The application logs all HCNetSDK operations:

```
INFO: Camera 1 [Entrance]: HCNetSDK worker started (DVR=192.168.1.100:8000, channel=1)
INFO: Camera 1: Attempting to login (attempt 1)
INFO: Successfully logged in to 192.168.1.100:8000 (device type: 0, channels: 8)
INFO: PlayCtrl decoder initialized on port 0
INFO: Started real play on channel 1 (play_handle: 12345)
INFO: Camera 1: 100 frames, 25.0 FPS
```

Check logs for:
- SDK initialization status
- Login success/failure
- Preview start/stop
- Frame decoding errors
- Reconnection attempts

## Migration from RTSP

To migrate existing RTSP cameras to HCNetSDK:

1. **Backup current configuration:**
   ```sql
   SELECT * FROM cameras WHERE source_type = 'rtsp';
   ```

2. **Update camera configuration:**
   ```sql
   UPDATE cameras 
   SET source_type = 'hcnetsdk',
       source_url = 'hcnetsdk://<dvr_ip>:8000@<user>:<pass>?channel=<ch>'
   WHERE id = <camera_id>;
   ```

3. **Restart the application:**
   ```bash
   # Stop the FastAPI server
   # Start the FastAPI server
   ```

4. **Verify in logs:**
   Look for HCNetSDK initialization messages and successful login.

## Performance Considerations

- HCNetSDK uses native C++ libraries, which are more efficient than RTSP
- CPU usage should be lower compared to RTSP decoding
- Network bandwidth usage is similar
- Latency is typically lower with direct SDK connection

## Security Notes

- DVR credentials are stored in the database in plain text
- Ensure your database is properly secured
- Use strong passwords for DVR accounts
- Limit DVR user permissions to only what's needed
- Consider using a dedicated DVR user for the application

## Support

For HCNetSDK-specific issues:
- Refer to Hikvision HCNetSDK documentation
- Check Hikvision developer forums
- Contact Hikvision technical support

For application integration issues:
- Check application logs
- Verify configuration format
- Test DVR connectivity independently
