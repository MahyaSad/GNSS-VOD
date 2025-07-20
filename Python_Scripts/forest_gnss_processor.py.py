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

def parse_forest_filename(filename):
    """Parse forest filename format: SEPT256k.20.sn1"""
    match = re.match(r'SEPT(\d{3})([a-z])\.20\.(\w+)', filename)
    if match:
        day = int(match.group(1))
        hour_code = match.group(2)
        ext = match.group(3)
        return day, hour_code, ext
    return None, None, None

def get_hour_number_from_code(hour_code):
    """Convert hour code to actual hour number"""
    hour_mapping = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
        'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
        'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17,
        's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23
    }
    return hour_mapping.get(hour_code.lower(), None)

def filter_gnss_no_sbas(satellite_list):
    """Filter GNSS satellites excluding SBAS"""
    gnss_satellites = []
    for sat in satellite_list:
        sat_str = str(sat).upper()
        
        # GPS satellites (G01-G32) - No SBAS
        if sat_str.startswith('G'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 32:  # GPS only, no SBAS
                    gnss_satellites.append(sat_str)
            except ValueError:
                continue
        
        # GLONASS satellites (R01-R24)
        elif sat_str.startswith('R'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 24:
                    gnss_satellites.append(sat_str)
            except ValueError:
                continue
        
        # Galileo satellites (E01-E36)
        elif sat_str.startswith('E'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 36:
                    gnss_satellites.append(sat_str)
            except ValueError:
                continue
        
        # BeiDou satellites (C01-C63)
        elif sat_str.startswith('C'):
            try:
                prn_num = int(sat_str[1:])
                if 1 <= prn_num <= 63:
                    gnss_satellites.append(sat_str)
            except ValueError:
                continue
        
        # Numeric format (assume GPS, still no SBAS)
        elif sat_str.isdigit():
            prn_num = int(sat_str)
            if 1 <= prn_num <= 32:  # Only GPS range
                gnss_satellites.append(f"G{prn_num:02d}")
    
    return sorted(list(set(gnss_satellites)))

def get_available_forest_data(base_path):
    """Scan forest directory to find all available days, hours, and data types"""
    base_path = Path(base_path)
    
    data_types = ['sn1', 'ele', 'azi']
    available_data = {}
    
    for data_type in data_types:
        type_path = base_path / data_type
        if type_path.exists():
            files = list(type_path.glob('SEPT*.20.*'))
            available_data[data_type] = []
            
            for file in files:
                day, hour_code, ext = parse_forest_filename(file.name)
                if day is not None and hour_code is not None:
                    hour_num = get_hour_number_from_code(hour_code)
                    if hour_num is not None:
                        available_data[data_type].append({
                            'day': day,
                            'hour_code': hour_code,
                            'hour_num': hour_num,
                            'filepath': file,
                            'filename': file.name
                        })
    
    return available_data

def create_datetime_from_components(df, base_year=2020):
    """Create datetime column from day_of_year and corrected_seconds"""
    base_date = pd.Timestamp(f'{base_year}-01-01')
    df['datetime'] = (
        base_date + 
        pd.to_timedelta(df['day_of_year'] - 1, unit='D') + 
        pd.to_timedelta(df['corrected_seconds'], unit='s')
    )
    return df

def load_forest_day_data_for_exact_match(day, base_path):
    """Load all hourly files for a specific day and correct timestamps for exact matching"""
    logger.info(f"Loading forest data for DOY {day} for exact time matching...")
    
    # Get available data for this day
    available_data = get_available_forest_data(base_path)
    
    # Filter for the specific day
    day_data = {}
    for data_type in ['sn1', 'ele', 'azi']:
        if data_type in available_data:
            day_files = [item for item in available_data[data_type] if item['day'] == day]
            day_data[data_type] = sorted(day_files, key=lambda x: x['hour_num'])
    
    if not all(data_type in day_data and day_data[data_type] for data_type in ['sn1', 'ele', 'azi']):
        logger.warning(f"DOY {day}: Missing data types")
        return {}
    
    # Load and combine all hourly files for each data type
    corrected_datasets = {}
    
    for data_type in ['sn1', 'ele', 'azi']:
        all_hourly_data = []
        
        for file_info in day_data[data_type]:
            hour_num = file_info['hour_num']
            filepath = file_info['filepath']
            
            # Load the file
            df = read_teqc_compact3_file_working(filepath)
            
            if df.empty:
                continue
            
            # Filter for GNSS satellites (excluding SBAS)
            all_satellites = df['satellite_id'].unique()
            gnss_satellites = filter_gnss_no_sbas(all_satellites)
            
            if not gnss_satellites:
                continue
            
            # Filter data for GNSS satellites
            satellite_data = df[df['satellite_id'].isin(gnss_satellites)].copy()
            
            if satellite_data.empty:
                continue
            
            # CRITICAL: Correct timestamps by adding hour offset
            satellite_data['corrected_seconds'] = satellite_data['seconds_of_day'] + (hour_num * 3600)
            satellite_data['day_of_year'] = day
            satellite_data['data_type'] = data_type
            
            all_hourly_data.append(satellite_data)
        
        if all_hourly_data:
            # Combine all hours and sort by corrected time
            combined_data = pd.concat(all_hourly_data, ignore_index=True)
            combined_data = combined_data.sort_values(['satellite_id', 'corrected_seconds'])
            
            # Remove duplicates (same satellite at same corrected time)
            combined_data = combined_data.drop_duplicates(['satellite_id', 'corrected_seconds'])
            
            # Create datetime column using corrected time
            combined_data = create_datetime_from_components(combined_data)
            
            corrected_datasets[data_type] = combined_data
            
            logger.info(f"{data_type.upper()}: {len(combined_data)} total observations")
        else:
            logger.warning(f"No {data_type} data loaded for DOY {day}")
            corrected_datasets[data_type] = pd.DataFrame()
    
    return corrected_datasets

def match_forest_satellite_data_exact_time(datasets, satellite_id, day):
    """Match elevation, azimuth, and SNR data for exact matching corrected timestamps in forest"""
    try:
        # Extract data for this satellite
        snr_data = datasets['sn1'][datasets['sn1']['satellite_id'] == satellite_id].copy()
        azi_data = datasets['azi'][datasets['azi']['satellite_id'] == satellite_id].copy()
        ele_data = datasets['ele'][datasets['ele']['satellite_id'] == satellite_id].copy()
        
        if snr_data.empty or azi_data.empty or ele_data.empty:
            return pd.DataFrame()
        
        # Sort by corrected time and remove duplicates
        snr_data = snr_data.sort_values('corrected_seconds').drop_duplicates('corrected_seconds')
        azi_data = azi_data.sort_values('corrected_seconds').drop_duplicates('corrected_seconds')
        ele_data = ele_data.sort_values('corrected_seconds').drop_duplicates('corrected_seconds')
        
        # Find common corrected timestamps across all three datasets
        snr_times = set(snr_data['corrected_seconds'].values)
        azi_times = set(azi_data['corrected_seconds'].values)
        ele_times = set(ele_data['corrected_seconds'].values)
        
        # Get intersection of all timestamps
        common_times = snr_times.intersection(azi_times).intersection(ele_times)
        
        if len(common_times) == 0:
            return pd.DataFrame()
        
        # Filter each dataset to common timestamps
        common_times_list = sorted(list(common_times))
        
        snr_filtered = snr_data[snr_data['corrected_seconds'].isin(common_times_list)].copy()
        azi_filtered = azi_data[azi_data['corrected_seconds'].isin(common_times_list)].copy()
        ele_filtered = ele_data[ele_data['corrected_seconds'].isin(common_times_list)].copy()
        
        # Merge the datasets on corrected_seconds
        # Start with SNR data as base
        result_df = snr_filtered.copy()
        result_df.rename(columns={'value': 'snr'}, inplace=True)
        
        # Merge azimuth data
        azi_merge = azi_filtered[['corrected_seconds', 'value']].rename(columns={'value': 'azimuth'})
        result_df = result_df.merge(azi_merge, on='corrected_seconds', how='inner')
        
        # Merge elevation data
        ele_merge = ele_filtered[['corrected_seconds', 'value']].rename(columns={'value': 'elevation'})
        result_df = result_df.merge(ele_merge, on='corrected_seconds', how='inner')
        
        # Ensure satellite ID is consistent
        result_df['satellite_id'] = satellite_id
        
        # Add time components for easier analysis
        result_df['hour'] = result_df['datetime'].dt.hour
        result_df['minute'] = result_df['datetime'].dt.minute
        result_df['second'] = result_df['datetime'].dt.second
        
        # Select final columns - using corrected_seconds instead of seconds_of_day
        final_columns = ['day_of_year', 'satellite_id', 'datetime', 'corrected_seconds', 
                        'hour', 'minute', 'second', 'azimuth', 'elevation', 'snr']
        
        return result_df[final_columns]
        
    except Exception as e:
        logger.error(f"Error matching forest satellite {satellite_id} data for DOY {day}: {e}")
        return pd.DataFrame()

def process_forest_single_day_data_exact_match(day_info, base_path):
    """Process data for a single day in forest - exact time matching only"""
    day = day_info['day']
    
    try:
        logger.info(f"Processing forest DOY {day} with exact time matching...")
        
        # Load data with time correction
        corrected_datasets = load_forest_day_data_for_exact_match(day, base_path)
        
        if len(corrected_datasets) < 3:
            logger.warning(f"DOY {day}: Missing required datasets")
            return pd.DataFrame()
        
        # Get all GNSS satellites available in SNR data
        snr_df = corrected_datasets['sn1']
        all_satellites = snr_df['satellite_id'].unique()
        gnss_satellites = filter_gnss_no_sbas(all_satellites)
        
        if len(gnss_satellites) == 0:
            logger.warning(f"DOY {day}: No GNSS satellites found")
            return pd.DataFrame()
        
        logger.info(f"DOY {day}: Found {len(gnss_satellites)} GNSS satellites: {gnss_satellites}")
        
        # Process each GNSS satellite with exact time matching
        all_matched_data = []
        
        for sat in gnss_satellites:
            sat_data = match_forest_satellite_data_exact_time(corrected_datasets, sat, day)
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
        logger.error(f"Error processing forest DOY {day}: {e}")
        return pd.DataFrame()

def process_forest_all_days_parallel_exact_match(base_path, max_workers=4):
    """Process all available forest days in parallel using exact time matching"""
    base_path = Path(base_path)
    
    # Get available data
    available_data = get_available_forest_data(base_path)
    
    if 'sn1' not in available_data:
        logger.error("No SNR data files found")
        return pd.DataFrame()
    
    # Get unique days that have SNR data
    available_days = sorted(list(set([item['day'] for item in available_data['sn1']])))
    
    logger.info(f"Found {len(available_days)} days with forest data: DOY {min(available_days)} to {max(available_days)}")
    
    # Process days in parallel
    all_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_day = {
            executor.submit(process_forest_single_day_data_exact_match, {'day': day}, base_path): day 
            for day in available_days
        }
        
        # Collect results with progress bar
        with tqdm(total=len(available_days), desc="Processing forest days (exact matching)") as pbar:
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

def process_forest_all_days_sequential_exact_match(base_path):
    """Process all available forest days sequentially using exact time matching"""
    base_path = Path(base_path)
    
    # Get available data
    available_data = get_available_forest_data(base_path)
    
    if 'sn1' not in available_data:
        logger.error("No SNR data files found")
        return pd.DataFrame()
    
    # Get unique days that have SNR data
    available_days = sorted(list(set([item['day'] for item in available_data['sn1']])))
    
    logger.info(f"Found {len(available_days)} days with forest data: DOY {min(available_days)} to {max(available_days)}")
    
    # Process days sequentially
    all_results = []
    
    for day in tqdm(available_days, desc="Processing forest days (exact matching)"):
        try:
            result = process_forest_single_day_data_exact_match({'day': day}, base_path)
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

def save_forest_exact_match_results(df, base_filename='Forest_GNSS_ExactMatch_Dataset'):
    """Save forest exact match results in multiple formats"""
    
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
        'environment': 'Forest (POLAR2_Forest)',
        'processing_method': 'exact_time_matching',
        'satellite_systems': 'GPS, GLONASS, Galileo, BeiDou (SBAS excluded)'
    }
    
    # Save as pickle with metadata
    pickle_filename = f"{base_filename}_{timestamp}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump({
            'data': df,
            'metadata': stats,
            'description': 'Forest GNSS dataset with exact time-matched elevation, azimuth, and SNR data for all GNSS satellites (GPS, GLONASS, Galileo, BeiDou - no SBAS)'
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
        f.write("Forest GNSS Exact Match Dataset Metadata\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved forest exact match results:")
    logger.info(f"  📊 Data: {pickle_filename} (pickle), {csv_filename} (CSV), {parquet_filename} (parquet)")
    logger.info(f"  📝 Metadata: {metadata_filename}")
    logger.info(f"  📈 Total exact matches: {len(df):,}")
    logger.info(f"  🛰️ Satellites: {len(df['satellite_id'].unique())}")
    logger.info(f"  📅 Days: {df['day_of_year'].nunique()}")

def create_forest_exact_match_summary_plots(df, save_dir='forest_exact_match_plots'):
    """Create summary plots for the forest exact match dataset"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. Satellite coverage by day
    plt.figure(figsize=(15, 8))
    pivot_data = df.groupby(['day_of_year', 'satellite_id']).size().unstack(fill_value=0)
    plt.imshow(pivot_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Number of exact matches')
    plt.xlabel('Day of Year')
    plt.ylabel('Satellite ID')
    plt.title('Forest Satellite Exact Matches by Day (GNSS - No SBAS)')
    plt.tight_layout()
    plt.savefig(save_dir / 'forest_satellite_exact_matches_by_day.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Daily exact match counts
    plt.figure(figsize=(15, 6))
    daily_counts = df.groupby('day_of_year').size()
    plt.plot(daily_counts.index, daily_counts.values, 'g-', alpha=0.7)
    plt.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='green')
    plt.xlabel('Day of Year')
    plt.ylabel('Number of Exact Matches')
    plt.title('Forest Daily Exact Match Counts (GNSS - No SBAS)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'forest_daily_exact_match_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. SNR distribution by satellite
    plt.figure(figsize=(20, 8))
    satellites = sorted(df['satellite_id'].unique())
    snr_data = [df[df['satellite_id'] == sat]['snr'].values for sat in satellites]
    plt.boxplot(snr_data, labels=satellites)
    plt.xlabel('Satellite ID')
    plt.ylabel('SNR (dB)')
    plt.title('Forest SNR Distribution by Satellite (Exact Matches - No SBAS)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'forest_snr_distribution_by_satellite_exact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Forest exact match summary plots saved to {save_dir}/")

def main_forest_exact_match_processing(base_path='unzip/gnss_data_archive/POLAR2_Forest/processed', 
                                     use_parallel=True, max_workers=4):
    """Main function to process all forest days and all GNSS satellites using EXACT TIME MATCHING ONLY"""
    
    logger.info("🌲 FOREST GNSS DATA PROCESSING - EXACT TIME MATCHING ONLY")
    logger.info("=" * 70)
    logger.info(f"Base path: {base_path}")
    logger.info(f"Processing method: EXACT TIME MATCHING (no interpolation)")
    logger.info(f"Satellite systems: GPS, GLONASS, Galileo, BeiDou (no SBAS)")
    logger.info(f"Parallel processing: {use_parallel} (workers: {max_workers})")
    
    start_time = time.time()
    
    # Process all days using exact time matching only
    if use_parallel:
        final_df = process_forest_all_days_parallel_exact_match(base_path, max_workers)
    else:
        final_df = process_forest_all_days_sequential_exact_match(base_path)
    
    processing_time = time.time() - start_time
    
    if not final_df.empty:
        logger.info(f"\n🎉 FOREST EXACT TIME MATCHING PROCESSING COMPLETE!")
        logger.info(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        logger.info(f"📊 Final dataset: {len(final_df):,} exact matches")
        logger.info(f"🛰️  Satellites: {final_df['satellite_id'].nunique()} ({sorted(final_df['satellite_id'].unique())})")
        logger.info(f"📅 Days: {final_df['day_of_year'].nunique()} (DOY {final_df['day_of_year'].min()}-{final_df['day_of_year'].max()})")
        
        # Save results
        save_forest_exact_match_results(final_df)
        
        # Create summary plots
        try:
            create_forest_exact_match_summary_plots(final_df)
        except Exception as e:
            logger.warning(f"Failed to create summary plots: {e}")
        
        return final_df
    else:
        logger.error("❌ No forest data processed successfully")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Process all forest data using EXACT TIME MATCHING ONLY
    forest_exact_data = main_forest_exact_match_processing(
        base_path='unzip/gnss_data_archive/POLAR2_Forest/processed',
        use_parallel=True,  # Set to False if you encounter issues
        max_workers=4
    )
    
    # Quick data preview
    if not forest_exact_data.empty:
        print("\n📋 Forest Exact Match Data Preview:")
        print(forest_exact_data.head(10))
        print(f"\n📊 Data Shape: {forest_exact_data.shape}")
        print(f"🏛️ Columns: {list(forest_exact_data.columns)}")
        
        # Show time range per satellite
        print(f"\n🛰️ Satellite Coverage:")
        for sat in sorted(forest_exact_data['satellite_id'].unique()):
            sat_data = forest_exact_data[forest_exact_data['satellite_id'] == sat]
            print(f"   {sat}: {len(sat_data)} exact matches, {sat_data['day_of_year'].nunique()} days")
            
        # Show example of matched data
        print(f"\n🔍 Example of exact matched data:")
        example_sat = forest_exact_data['satellite_id'].iloc[0]
        example_day = forest_exact_data['day_of_year'].iloc[0]
        example_data = forest_exact_data[
            (forest_exact_data['satellite_id'] == example_sat) & 
            (forest_exact_data['day_of_year'] == example_day)
        ].head(5)
        print(example_data[['satellite_id', 'corrected_seconds', 'azimuth', 'elevation', 'snr']])