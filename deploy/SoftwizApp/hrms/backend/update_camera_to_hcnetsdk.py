"""
Update camera configuration to use HCNetSDK instead of RTSP.
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, text
from app.core.config import get_settings

settings = get_settings()

# Create database connection
engine = create_engine(settings.database_url)

# First, check current configuration
check_sql = text("""
    SELECT id, name, source_type, source_url, enabled 
    FROM cameras 
    WHERE id = 7
""")

try:
    with engine.connect() as conn:
        # Check current config
        print("Current camera configuration:")
        result = conn.execute(check_sql)
        for row in result:
            print(f"  Camera {row.id} ({row.name}): source_type={row.source_type}, source_url={row.source_url}, enabled={row.enabled}")
        
        # Update camera 7 to use HCNetSDK and enable it
        update_sql = text("""
            UPDATE cameras 
            SET source_type = 'hcnetsdk',
                source_url = 'hcnetsdk://192.168.29.181:8000@anilchanna:test@123?channel=1',
                enabled = true
            WHERE id = 7
        """)
        
        result = conn.execute(update_sql)
        conn.commit()
        print(f"\nUpdated {result.rowcount} camera(s)")
        print("Camera 5 now configured to use HCNetSDK")
        
        # Verify the update
        print("\nUpdated configuration:")
        result = conn.execute(check_sql)
        for row in result:
            print(f"  Camera {row.id} ({row.name}): source_type={row.source_type}, source_url={row.source_url}, enabled={row.enabled}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
