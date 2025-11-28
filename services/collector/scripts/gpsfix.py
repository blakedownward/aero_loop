#!/usr/bin/env python3
"""
GPS Fix Script

Reads GPS data from stdin (via gpspipe -w) and waits for a valid GPS fix.
Exits when a 3D fix is acquired.
"""

import sys
import json

def main():
    """Read GPS data from stdin and wait for 3D fix."""
    fix_acquired = False
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            # Check for TPV (Time Position Velocity) class messages
            if data.get('class') == 'TPV':
                mode = data.get('mode')
                
                # Mode 3 = 3D fix (has latitude, longitude, and altitude)
                if mode == 3:
                    lat = data.get('lat')
                    lon = data.get('lon')
                    
                    if lat is not None and lon is not None:
                        print(f"GPS 3D fix acquired: lat={lat}, lon={lon}", file=sys.stderr)
                        fix_acquired = True
                        break
                elif mode == 2:
                    # Mode 2 = 2D fix (has lat/lon but no altitude)
                    lat = data.get('lat')
                    lon = data.get('lon')
                    if lat is not None and lon is not None:
                        print(f"GPS 2D fix acquired: lat={lat}, lon={lon}", file=sys.stderr)
                        # Continue waiting for 3D fix, or use 2D if acceptable
                        # For now, we'll accept 2D fix as well
                        fix_acquired = True
                        break
        
        except json.JSONDecodeError:
            # Skip invalid JSON lines
            continue
        except Exception as e:
            print(f"Error processing GPS data: {e}", file=sys.stderr)
            continue
    
    if not fix_acquired:
        print("Warning: GPS fix not acquired before timeout", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(0)

if __name__ == '__main__':
    main()

