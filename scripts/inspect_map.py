
from nuscenes.map_expansion.map_api import NuScenesMap
import os

# Use the specific map file location if needed, but NuScenesMap expects dataroot + map_name
# We will just use the standard init but point to the dataroot
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
map_name = "singapore-onenorth"

try:
    nusc_map = NuScenesMap(dataroot=NUSCENES_ROOT, map_name=map_name)
    print("NuScenesMap attributes:", dir(nusc_map))
    

    # Get a random lane to test


    # Check connectivity
    if hasattr(nusc_map, 'connectivity'):
        c = nusc_map.connectivity
        if c:
            key = list(c.keys())[0]
            print(f"Connectivity for {key}: {c[key]}")

    # Get a random connection to test
    records = nusc_map.get_records_in_radius(1000, 1000, 100, ['lane_connector'])
    if records['lane_connector']:
        lc_token = records['lane_connector'][0]
        lc_record = nusc_map.get('lane_connector', lc_token)
        print("Lane Connector keys:", lc_record.keys())
        pass
        

    # Print ALL attributes
    print("All NuScenesMap attributes:")
    for m in dir(nusc_map):
        if not m.startswith('_'): # Skip private
            print(m)

        
except Exception as e:
    print(f"Error: {e}")
