"""
Sentinel-2 Time Series Analysis with Multiple Vegetation Indices
Python implementation - FIXED VERSION for large parcel collections
"""

import ee
import time

# Initialize Earth Engine
PROJECT_ID = 'ee-atikul'

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"✓ Earth Engine initialized successfully with project: {PROJECT_ID}")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're authenticated: earthengine authenticate")
    print("2. Check project is registered at: https://code.earthengine.google.com/")
    print(f"3. Verify project ID is correct: {PROJECT_ID}")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

TABLE_NAME = "projects/ee-atikul/assets/Wheat_2017"
START_DATE = '2016-09-01'
END_DATE = '2017-10-31'

# TESTING MODE - Set to True to test with limited parcels
TESTING_MODE = True  # Set to False to process all parcels
TEST_PARCEL_LIMIT = 3  # Number of parcels to test with

# Cloud filtering parameters - SIMPLIFIED to avoid timeouts
MAX_CLOUD_PERCENT = 70  # Reduced for faster processing

# Export configuration
EXPORT_FOLDER = 'Phen_Ger_new'
EXPORT_PREFIX = 'wheat17'

# Bands and VIs to process
BANDS_AND_VIS = [
    'EVI', 'NDRE', 'CIrededge', 'MTCI',
    'blue', 'green', 'red', 're1', 're2', 're3',
    'nir', 'swir1', 'swir2'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_date(img):
    """Add date as property."""
    date = img.date().format('YYYYMMdd')
    return img.set('date', ee.Number.parse(date))


# ============================================================================
# VEGETATION INDICES FUNCTIONS
# ============================================================================

def add_evi(img):
    """Add Enhanced Vegetation Index."""
    evi = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': img.select('nir'),
            'RED': img.select('red'),
            'BLUE': img.select('blue')
        }
    )
    return img.addBands(evi.float().rename('EVI'))


def add_ndre(img):
    """Add Normalized Difference Red Edge."""
    ndre = img.normalizedDifference(['nir', 're1'])
    return img.addBands(ndre.float().rename('NDRE'))


def add_ci_red_edge(img):
    """Add Chlorophyll Index Red Edge."""
    cire = img.expression(
        '(NIR / RE1) - 1',
        {
            'NIR': img.select('nir'),
            'RE1': img.select('re1')
        }
    )
    return img.addBands(cire.float().rename('CIrededge'))


def add_mtci(img):
    """Add MERIS Terrestrial Chlorophyll Index."""
    mtci = img.expression(
        '(RE2 - RE1) / (RE1 - RED)',
        {
            'RE2': img.select('re2'),
            'RE1': img.select('re1'),
            'RED': img.select('red')
        }
    )
    return img.addBands(mtci.float().rename('MTCI'))


def add_all_vis(img):
    """Add all vegetation indices."""
    img = add_evi(img)
    img = add_ndre(img)
    img = add_ci_red_edge(img)
    img = add_mtci(img)
    return img


# ============================================================================
# IMAGE COLLECTION PROCESSING - SIMPLIFIED
# ============================================================================

def get_processed_collection(parcel_bounds):
    """Get processed Sentinel-2 collection for given bounds."""
    # Use S2_HARMONIZED for consistency across time
    print("  Filtering Sentinel-2 collection by date and cloud cover...")
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(START_DATE, END_DATE) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', MAX_CLOUD_PERCENT) \
        .filterBounds(parcel_bounds) \
        .sort('system:time_start') \
        .map(add_date)
    
    print("  Processing and scaling bands...")
    
    def process_image(img):
        # Rescale to 0-1
        t = img.select([
            'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'
        ]).divide(10000)
        
        # Rename bands
        t = t.select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'],
            ['blue', 'green', 'red', 're1', 're2', 're3', 'nir', 'swir1', 'swir2']
        )
        
        out = t.copyProperties(img).copyProperties(img, ['system:time_start'])
        return out
    
    s2_processed = s2.map(process_image).map(add_all_vis)
    
    return s2_processed


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def format_to_wide_table(triplets, parcel_fc):
    """
    Convert long format to wide format with dates as columns.
    This is the CRITICAL function that formats data correctly.
    """
    # Get distinct parcels
    distinct_xids = triplets.distinct('XID')
    
    def pivot_parcel(parcel_feature):
        xid = parcel_feature.get('XID')
        
        # Get all records for this parcel
        parcel_records = triplets.filter(ee.Filter.eq('XID', xid))
        
        # Convert to dictionary of date: value pairs
        def create_date_dict(feat, prev_dict):
            date = ee.Feature(feat).get('date')
            value = ee.Feature(feat).get('value')
            return ee.Dictionary(prev_dict).set(date, value)
        
        date_dict = ee.Dictionary(
            parcel_records.iterate(create_date_dict, ee.Dictionary())
        )
        
        return parcel_feature.set(date_dict)
    
    wide_table = distinct_xids.map(pivot_parcel)
    
    # Fill missing dates with -9999
    year = int(START_DATE.split('-')[0])
    month = int(START_DATE.split('-')[1])
    day = int(START_DATE.split('-')[2])
    
    keys = ee.List.sequence(0, 364).map(
        lambda d: ee.Date.fromYMD(year, month, day).advance(d, 'day').format('YYYYMMdd')
    )
    values = ee.List.repeat(-9999, 365)
    empty_dict = ee.Dictionary.fromLists(keys, values)
    
    def fill_missing(f):
        return f.set(empty_dict.combine(f.toDictionary()))
    
    wide_table = wide_table.map(fill_missing)
    
    return wide_table


def extract_band_values(parcel_fc, band_name, collection):
    """Extract band/VI values for all parcels."""
    print(f"  Extracting {band_name} values...")
    
    def extract_for_image(img):
        date_str = ee.Date(img.get('system:time_start')).format('YYYYMMdd')
        
        reduced = img.select(band_name).reduceRegions(
            collection=parcel_fc.select(['XID']),
            reducer=ee.Reducer.mean(),
            scale=10,
            tileScale=4
        )
        
        def add_metadata(f):
            value = f.get('mean')
            # Handle nulls
            value = ee.Algorithms.If(
                ee.Algorithms.IsEqual(value, None),
                -9999,
                value
            )
            return f.select(['XID']).set({
                'value': value,
                'date': date_str
            })
        
        return reduced.map(add_metadata)
    
    triplets = collection.map(extract_for_image).flatten()
    
    return triplets


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("Sentinel-2 VI Extraction - Simplified Python Version")
    print("="*70)
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    # Load parcel collection
    print(f"\nLoading parcels from: {TABLE_NAME}")
    parcel = ee.FeatureCollection(TABLE_NAME)
    
    # Apply testing limit if enabled
    if TESTING_MODE:
        print(f"\n⚠️  TESTING MODE ENABLED")
        print(f"   Processing only first {TEST_PARCEL_LIMIT} parcels")
        parcel = parcel.limit(TEST_PARCEL_LIMIT)
    else:
        total_parcels = parcel.size().getInfo()
        print(f"Processing ALL {total_parcels} parcels")
    
    # Get parcel bounds for filtering imagery
    # Use convex hull or bounding box to avoid "too many edges" error
    print("\nCalculating study area bounds...")
    parcel_bounds = parcel.geometry().convexHull(maxError=100)  # Simplified boundary
    print("Study area bounds calculated successfully")
    
    # Get processed collection
    print("\nPreparing Sentinel-2 image collection...")
    collection = get_processed_collection(parcel_bounds)
    
    # Get number of images without triggering the error
    print("Counting available images...")
    try:
        num_images = collection.size().getInfo()
        print(f"Found {num_images} images in date range")
        
        if num_images == 0:
            print("ERROR: No images found! Check your date range and parcel location.")
            return
    except Exception as e:
        print(f"Note: Could not get exact image count (this is OK)")
        print(f"Proceeding with export anyway...")
        num_images = "Unknown"
    
    # Process each band/VI
    print(f"\nProcessing {len(BANDS_AND_VIS)} bands/VIs...")
    print("="*70)
    
    all_tasks = []
    
    for i, band_name in enumerate(BANDS_AND_VIS, 1):
        print(f"\n[{i}/{len(BANDS_AND_VIS)}] Processing {band_name}...")
        
        try:
            # Extract values
            triplets = extract_band_values(parcel, band_name, collection)
            
            # Format to wide table
            print(f"  Formatting to wide table...")
            wide_table = format_to_wide_table(triplets, parcel)
            
            # Create task name
            if TESTING_MODE:
                task_name = f"{band_name}_{EXPORT_PREFIX}_TEST"
            else:
                task_name = f"{band_name}_{EXPORT_PREFIX}"
            
            # Create export task
            task = ee.batch.Export.table.toDrive(
                collection=wide_table,
                description=task_name,
                folder=EXPORT_FOLDER,
                fileNamePrefix=task_name,
                fileFormat='CSV'
            )
            
            all_tasks.append((task, task_name))
            print(f"  ✓ Task created: {task_name}")
            
        except Exception as e:
            print(f"  ✗ ERROR creating task for {band_name}: {e}")
            continue
    
    # Start all tasks
    print("\n" + "="*70)
    print(f"Starting {len(all_tasks)} export tasks...")
    print("="*70)
    
    for i, (task, task_name) in enumerate(all_tasks, 1):
        try:
            task.start()
            print(f"  [{i}/{len(all_tasks)}] Started: {task_name}")
            time.sleep(1)  # Delay to avoid rate limiting
        except Exception as e:
            print(f"  [{i}/{len(all_tasks)}] ERROR starting {task_name}: {e}")
    
    print("\n" + "="*70)
    print(f"✓ Successfully started {len(all_tasks)} tasks!")
    print("="*70)
    
    if TESTING_MODE:
        print(f"\n⚠️  TESTING MODE: Only {TEST_PARCEL_LIMIT} parcels processed")
        print("   If results look good, set TESTING_MODE = False and run again")
    
    print("\nMonitor progress at: https://code.earthengine.google.com/tasks")
    print(f"Files will be saved to Google Drive folder: {EXPORT_FOLDER}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    if TESTING_MODE:
        print("1. Wait for test exports to complete")
        print("2. Check the CSV files in Google Drive")
        print("3. If data looks correct:")
        print("   - Set TESTING_MODE = False")
        print("   - Run script again to process all parcels")
    else:
        print("1. Wait for all exports to complete")
        print("2. Download CSV files from Google Drive")
        print("3. Verify data quality")


if __name__ == "__main__":
    main()