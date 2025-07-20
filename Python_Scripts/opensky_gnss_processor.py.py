import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pickle
import warnings
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_filename_official_format(filename):
    """Parse POLAR filenames to extract day and identifier"""
    match = re.match(r'SEPT(\d{3})(\d)\.20\.(\w+)', filename)
    if match:
        day = int(match.group(1))
        identifier = match.group(2)
        ext = match.group(3)
        return day, identifier, ext
    return None, None, None

def create_datetime_from_components(df, base_year=2020):
    """Create datetime column from day_of_year and seconds_of_day"""
    base_date = pd.Timestamp(f'{base_year}-01-01')
    df['datetime'] = (
        base_date + 
        pd.to_timedelta(df['day_of_year'] - 1, unit='D') + 
        pd.to_timedelta(df['seconds_of_day'], unit='s')
    )
    return df

def read_teqc_compact3_file_working(filepath):
    """Read TEQC file and extract data"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return pd.DataFrame()
        
        data_rows = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith(('COMPACT', 'GPS_START', '#')):
                i += 1
                continue
            
            parts = line.split()
            if len(parts) < 2:
                i += 1
                continue
            
            try:
                time_val = float(parts[0])
                if not (0 <= time_val <= 86400):
                    i += 1
                    continue
                
                if len(parts) >= 3:
                    try:
                        sat_count = int(parts[1])
                        if sat_count > 0 and len(parts) >= sat_count + 2:
                            satellites = parts[2:2+sat_count]
                            
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if next_line and not next_line.startswith(('COMPACT', 'GPS_START')):
                                    value_parts = next_line.split()
                                    try:
                                        values = [float(v) for v in value_parts]
                                        
                                        for j, sat in enumerate(satellites):
                                            if j < len(values) and values[j] != -1.0:
                                                data_rows.append({
                                                    'seconds_of_day': time_val,
                                                    'satellite_id': sat,
                                                    'value': values[j]
                                                })
                                        i += 2
                                        continue
                                    except ValueError:
                                        pass
                    except ValueError:
                        pass
                
                i += 1
                
            except ValueError:
                i += 1
                continue
        
        return pd.DataFrame(data_rows)
        
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def get_available_days_and_files(base_path):
    """Scan directory to find all available days and data types"""
    base_path = Path(base_path)
    
    # Check for different data types
    data_types = ['ele', 'azi', 'sn1']  # elevation, azimuth, SNR
    available_data = {}
    
    for data_type in data_types:
        type_path = base_path / data_type
        if type_path.exists():
            files = list(type_path.glob('SEPT*.20.*'))
            available_data[data_type] = []
            
            for file in files:
                day, identifier, ext = parse_filename_official_format(file.name)
                if day is not None:
                    available_data[data_type].append({
                        'day': day,
                        'identifier': identifier,
                        'filepath': file,
                        'filename': file.name
                    })
    
    return available_data

def filter_all_satellites_including_sbas(satellite_list):
    """Filter ALL satellites including SBAS - no filtering, just standardize format"""
    all_satellites = []
    for sat in satellite_list:
        sat_str = str(sat).upper()
        
        # GPS satellites (G01-G32) + SBAS (G33-G64, includes WAAS, EGNOS, etc.)
        if sat_str.startswith('G'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 64:  # Include all GPS and SBAS
                    all_satellites.append(sat_str)
            except ValueError:
                continue
        
        # GLONASS satellites (R01-R24)
        elif sat_str.startswith('R'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 24:
                    all_satellites.append(sat_str)
            except ValueError:
                continue
        
        # Galileo satellites (E01-E36)
        elif sat_str.startswith('E'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 36:
                    all_satellites.append(sat_str)
            except ValueError:
                continue
        
        # BeiDou satellites (C01-C63)
        elif sat_str.startswith('C'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 63:
                    all_satellites.append(sat_str)
            except ValueError:
                continue
        
        # SBAS satellites (S01-S64) - if using S prefix
        elif sat_str.startswith('S'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 64:
                    all_satellites.append(sat_str)
            except ValueError:
                continue
        
        # Numeric format - include ALL ranges
        elif sat_str.isdigit():
            prn_num = int(sat_str)
            if 1 <= prn_num <= 64:  # Extended range to include SBAS
                # Try to determine if it's SBAS based on PRN number
                if 33 <= prn_num <= 64:
                    all_satellites.append(f"G{prn_num:02d}")  # SBAS satellites
                elif 1 <= prn_num <= 32:
                    all_satellites.append(f"G{prn_num:02d}")  # GPS satellites
        
        # Keep any other format as-is (in case there are other naming conventions)
        else:
            all_satellites.append(sat_str)
    
    return sorted(list(set(all_satellites)))

def match_satellite_data_exact_time(datasets, satellite_id, day):
    """Match elevation, azimuth, and SNR data for exact matching timestamps"""
    try:
        # Extract data for this satellite
        snr_data = datasets['sn1'][datasets['sn1']['satellite_id'] == satellite_id].copy()
        azi_data = datasets['azi'][datasets['azi']['satellite_id'] == satellite_id].copy()
        ele_data = datasets['ele'][datasets['ele']['satellite_id'] == satellite_id].copy()
        
        if snr_data.empty or azi_data.empty or ele_data.empty:
            return pd.DataFrame()
        
        # Sort by time and remove duplicates
        snr_data = snr_data.sort_values('seconds_of_day').drop_duplicates('seconds_of_day')
        azi_data = azi_data.sort_values('seconds_of_day').drop_duplicates('seconds_of_day')
        ele_data = ele_data.sort_values('seconds_of_day').drop_duplicates('seconds_of_day')
        
        # Find common timestamps across all three datasets
        snr_times = set(snr_data['seconds_of_day'].values)
        azi_times = set(azi_data['seconds_of_day'].values)
        ele_times = set(ele_data['seconds_of_day'].values)
        
        # Get intersection of all timestamps
        common_times = snr_times.intersection(azi_times).intersection(ele_times)
        
        if len(common_times) == 0:
            return pd.DataFrame()
        
        # Filter each dataset to common timestamps
        common_times_list = sorted(list(common_times))
        
        snr_filtered = snr_data[snr_data['seconds_of_day'].isin(common_times_list)].copy()
        azi_filtered = azi_data[azi_data['seconds_of_day'].isin(common_times_list)].copy()
        ele_filtered = ele_data[ele_data['seconds_of_day'].isin(common_times_list)].copy()
        
        # Merge the datasets on seconds_of_day
        # Start with SNR data as base
        result_df = snr_filtered.copy()
        result_df.rename(columns={'value': 'snr'}, inplace=True)
        
        # Merge azimuth data
        azi_merge = azi_filtered[['seconds_of_day', 'value']].rename(columns={'value': 'azimuth'})
        result_df = result_df.merge(azi_merge, on='seconds_of_day', how='inner')
        
        # Merge elevation data
        ele_merge = ele_filtered[['seconds_of_day', 'value']].rename(columns={'value': 'elevation'})
        result_df = result_df.merge(ele_merge, on='seconds_of_day', how='inner')
        
        # Ensure satellite ID is consistent
        result_df['satellite_id'] = satellite_id
        
        # Add time components for easier analysis
        result_df['hour'] = result_df['datetime'].dt.hour
        result_df['minute'] = result_df['datetime'].dt.minute
        result_df['second'] = result_df['datetime'].dt.second
        
        # Select final columns
        final_columns = ['day_of_year', 'satellite_id', 'datetime', 'seconds_of_day', 
                        'hour', 'minute', 'second', 'azimuth', 'elevation', 'snr']
        
        return result_df[final_columns]
        
    except Exception as e:
        logger.error(f"Error matching satellite {satellite_id} data for DOY {day}: {e}")
        return pd.DataFrame()

def process_single_day_data_exact_match(day_info, base_path):
    """Process data for a single day - returns exactly matched data (no interpolation)"""
    day = day_info['day']
    
    try:
        logger.info(f"Processing DOY {day} with exact time matching...")
        
        # Load data files for this day
        data_files = {}
        for data_type in ['ele', 'azi', 'sn1']:
            filepath = base_path / data_type / f'SEPT{day:03d}0.20.{data_type}'
            if filepath.exists():
                data_files[data_type] = filepath
        
        # Check if we have all required files
        if len(data_files) < 3:
            logger.warning(f"DOY {day}: Missing data files. Found: {list(data_files.keys())}")
            return pd.DataFrame()
        
        # Load each data type
        datasets = {}
        for data_type, filepath in data_files.items():
            df = read_teqc_compact3_file_working(filepath)
            
            if df.empty:
                logger.warning(f"DOY {day}: No data in {data_type} file")
                continue
            
            # Add metadata
            df['day_of_year'] = day
            df['data_type'] = data_type
            df = create_datetime_from_components(df)
            
            datasets[data_type] = df
        
        if len(datasets) < 3:
            logger.warning(f"DOY {day}: Failed to load all datasets")
            return pd.DataFrame()
        
        # Get ALL satellites available in SNR data (including SBAS)
        snr_df = datasets['sn1']
        all_satellites = snr_df['satellite_id'].unique()
        all_satellites_filtered = filter_all_satellites_including_sbas(all_satellites)
        
        if len(all_satellites_filtered) == 0:
            logger.warning(f"DOY {day}: No satellites found")
            return pd.DataFrame()
        
        logger.info(f"DOY {day}: Found {len(all_satellites_filtered)} satellites (including SBAS): {all_satellites_filtered}")
        
        # Process each satellite (including SBAS) with exact time matching
        all_matched_data = []
        
        for sat in all_satellites_filtered:
            sat_data = match_satellite_data_exact_time(datasets, sat, day)
            if not sat_data.empty:
                all_matched_data.append(sat_data)
        
        if all_matched_data:
            combined_df = pd.concat(all_matched_data, ignore_index=True)
            logger.info(f"DOY {day}: Successfully matched {len(all_matched_data)} satellites, {len(combined_df)} total exact matches")
            return combined_df
        else:
            logger.warning(f"DOY {day}: No successful satellite matching")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing DOY {day}: {e}")
        return pd.DataFrame()

def process_all_days_parallel_exact_match(base_path, max_workers=4):
    """Process all available days in parallel using exact time matching"""
    base_path = Path(base_path)
    
    # Get available data
    available_data = get_available_days_and_files(base_path)
    
    if 'sn1' not in available_data:
        logger.error("No SNR data files found")
        return pd.DataFrame()
    
    # Get unique days that have SNR data
    available_days = sorted(list(set([item['day'] for item in available_data['sn1']])))
    
    logger.info(f"Found {len(available_days)} days with data: DOY {min(available_days)} to {max(available_days)}")
    
    # Process days in parallel
    all_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_day = {
            executor.submit(process_single_day_data_exact_match, {'day': day}, base_path): day 
            for day in available_days
        }
        
        # Collect results with progress bar
        with tqdm(total=len(available_days), desc="Processing days (exact matching)") as pbar:
            for future in as_completed(future_to_day):
                day = future_to_day[future]
                try:
                    result = future.result()
                    if not result.empty:
                        all_results.append(result)
                    pbar.set_postfix(day=f"DOY{day}", processed=len(all_results))
                except Exception as e:
                    logger.error(f"DOY {day} failed: {e}")
                finally:
                    pbar.update(1)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Successfully processed {len(all_results)} days, total {len(final_df)} exact matches")
        return final_df
    else:
        logger.error("No successful processing results")
        return pd.DataFrame()

def process_all_days_sequential_exact_match(base_path):
    """Process all available days sequentially using exact time matching"""
    base_path = Path(base_path)
    
    # Get available data
    available_data = get_available_days_and_files(base_path)
    
    if 'sn1' not in available_data:
        logger.error("No SNR data files found")
        return pd.DataFrame()
    
    # Get unique days that have SNR data
    available_days = sorted(list(set([item['day'] for item in available_data['sn1']])))
    
    logger.info(f"Found {len(available_days)} days with data: DOY {min(available_days)} to {max(available_days)}")
    
    # Process days sequentially
    all_results = []
    
    for day in tqdm(available_days, desc="Processing days (exact matching)"):
        try:
            result = process_single_day_data_exact_match({'day': day}, base_path)
            if not result.empty:
                all_results.append(result)
                logger.info(f"DOY {day}: {len(result)} exact matches processed")
        except Exception as e:
            logger.error(f"DOY {day} failed: {e}")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Successfully processed {len(all_results)} days, total {len(final_df)} exact matches")
        return final_df
    else:
        logger.error("No successful processing results")
        return pd.DataFrame()

def save_exact_match_results(df, base_filename='OpenSky_GNSS_ExactMatch_Dataset'):
    """Save exact match results in multiple formats with comprehensive metadata"""
    
    if df.empty:
        logger.error("No data to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics
    stats = {
        'total_observations': len(df),
        'unique_days': df['day_of_year'].nunique(),
        'unique_satellites': df['satellite_id'].nunique(),
        'day_range': f"DOY {df['day_of_year'].min()} to {df['day_of_year'].max()}",
        'satellite_list': sorted(df['satellite_id'].unique().tolist()),
        'time_span': f"{df['datetime'].min()} to {df['datetime'].max()}",
        'processing_date': datetime.now(),
        'data_columns': df.columns.tolist(),
        'snr_range': f"{df['snr'].min():.1f} to {df['snr'].max():.1f} dB",
        'elevation_range': f"{df['elevation'].min():.1f}° to {df['elevation'].max():.1f}°",
        'azimuth_range': f"{df['azimuth'].min():.1f}° to {df['azimuth'].max():.1f}°",
        'processing_method': 'exact_time_matching',
        'environment': 'Open-sky (POLAR1_Roof)'
    }
    
    # Save as pickle with metadata
    pickle_filename = f"{base_filename}_{timestamp}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump({
            'data': df,
            'metadata': stats,
            'description': 'Open-sky satellite dataset with exact time-matched elevation, azimuth, and SNR data for ALL satellites including SBAS'
        }, f)
    
    # Save as CSV
    csv_filename = f"{base_filename}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Save as parquet for better performance
    parquet_filename = f"{base_filename}_{timestamp}.parquet"
    df.to_parquet(parquet_filename, index=False)
    
    # Save metadata as text file
    metadata_filename = f"{base_filename}_metadata_{timestamp}.txt"
    with open(metadata_filename, 'w') as f:
        f.write("Open-Sky GNSS Exact Match Dataset Metadata\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved exact match results:")
    logger.info(f"  📊 Data: {pickle_filename} (pickle), {csv_filename} (CSV), {parquet_filename} (parquet)")
    logger.info(f"  📝 Metadata: {metadata_filename}")
    logger.info(f"  📈 Total exact matches: {len(df):,}")
    logger.info(f"  🛰️ Satellites: {len(df['satellite_id'].unique())}")
    logger.info(f"  📅 Days: {df['day_of_year'].nunique()}")

def create_exact_match_summary_plots(df, save_dir='opensky_exact_match_plots'):
    """Create summary plots for the exact match dataset"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. Satellite coverage by day
    plt.figure(figsize=(15, 8))
    pivot_data = df.groupby(['day_of_year', 'satellite_id']).size().unstack(fill_value=0)
    plt.imshow(pivot_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Number of exact matches')
    plt.xlabel('Day of Year')
    plt.ylabel('Satellite ID')
    plt.title('Open-Sky Satellite Exact Matches by Day')
    plt.tight_layout()
    plt.savefig(save_dir / 'opensky_satellite_exact_matches_by_day.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Daily exact match counts
    plt.figure(figsize=(15, 6))
    daily_counts = df.groupby('day_of_year').size()
    plt.plot(daily_counts.index, daily_counts.values, 'b-', alpha=0.7)
    plt.fill_between(daily_counts.index, daily_counts.values, alpha=0.3)
    plt.xlabel('Day of Year')
    plt.ylabel('Number of Exact Matches')
    plt.title('Open-Sky Daily Exact Match Counts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'opensky_daily_exact_match_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. SNR distribution by satellite
    plt.figure(figsize=(20, 8))
    satellites = sorted(df['satellite_id'].unique())
    snr_data = [df[df['satellite_id'] == sat]['snr'].values for sat in satellites]
    plt.boxplot(snr_data, labels=satellites)
    plt.xlabel('Satellite ID')
    plt.ylabel('SNR (dB)')
    plt.title('Open-Sky SNR Distribution by Satellite (Exact Matches)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'opensky_snr_distribution_by_satellite_exact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Exact match summary plots saved to {save_dir}/")

def main_opensky_exact_match_processing(base_path='unzip/gnss_data_archive/POLAR1_Roof/processed', 
                                      use_parallel=True, max_workers=4):
    """Main function to process all days and ALL satellites (including SBAS) using EXACT TIME MATCHING ONLY"""
    
    logger.info("🚀 OPEN-SKY SATELLITE DATA PROCESSING - EXACT TIME MATCHING (ALL SATELLITES INCLUDING SBAS)")
    logger.info("=" * 90)
    logger.info(f"Base path: {base_path}")
    logger.info(f"Processing method: EXACT TIME MATCHING (no interpolation)")
    logger.info(f"Satellite types: ALL (GPS, GLONASS, Galileo, BeiDou, SBAS)")
    logger.info(f"Parallel processing: {use_parallel} (workers: {max_workers})")
    
    start_time = time.time()
    
    # Process all days using exact time matching only
    if use_parallel:
        final_df = process_all_days_parallel_exact_match(base_path, max_workers)
    else:
        final_df = process_all_days_sequential_exact_match(base_path)
    
    processing_time = time.time() - start_time
    
    if not final_df.empty:
        logger.info(f"\n🎉 EXACT TIME MATCHING PROCESSING COMPLETE!")
        logger.info(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        logger.info(f"📊 Final dataset: {len(final_df):,} exact matches")
        logger.info(f"🛰️  Satellites: {final_df['satellite_id'].nunique()} ({sorted(final_df['satellite_id'].unique())})")
        logger.info(f"📅 Days: {final_df['day_of_year'].nunique()} (DOY {final_df['day_of_year'].min()}-{final_df['day_of_year'].max()})")
        
        # Save results
        save_exact_match_results(final_df)
        
        # Create summary plots
        try:
            create_exact_match_summary_plots(final_df)
        except Exception as e:
            logger.warning(f"Failed to create summary plots: {e}")
        
        return final_df
    else:
        logger.error("❌ No data processed successfully")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Process all data using EXACT TIME MATCHING (ALL SATELLITES INCLUDING SBAS)
    opensky_exact_data = main_opensky_exact_match_processing(
        base_path='unzip/gnss_data_archive/POLAR1_Roof/processed',
        use_parallel=True,  # Set to False if you encounter issues
        max_workers=4
    )
    
    # Quick data preview
    if not opensky_exact_data.empty:
        print("\n📋 Open-Sky Exact Match Data Preview:")
        print(opensky_exact_data.head(10))
        print(f"\n📊 Data Shape: {opensky_exact_data.shape}")
        print(f"🏛️ Columns: {list(opensky_exact_data.columns)}")
        
        # Show time range per satellite
        print(f"\n🛰️ Satellite Coverage:")
        for sat in sorted(opensky_exact_data['satellite_id'].unique()):
            sat_data = opensky_exact_data[opensky_exact_data['satellite_id'] == sat]
            print(f"   {sat}: {len(sat_data)} exact matches, {sat_data['day_of_year'].nunique()} days")
            
        # Show example of matched data
        print(f"\n🔍 Example of exact matched data:")
        example_sat = opensky_exact_data['satellite_id'].iloc[0]
        example_day = opensky_exact_data['day_of_year'].iloc[0]
        example_data = opensky_exact_data[
            (opensky_exact_data['satellite_id'] == example_sat) & 
            (opensky_exact_data['day_of_year'] == example_day)
        ].head(5)
        print(example_data[['satellite_id', 'seconds_of_day', 'azimuth', 'elevation', 'snr']])