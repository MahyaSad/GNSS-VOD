#!/usr/bin/env python3
"""
ENHANCED GNSS MATCHER WITH COMPLETE TIMESTAMP OUTPUT
==================================================
Preserves day-of-year, full datetime, and all timing details in output
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedGNSSMatcher:
    """Enhanced GNSS matcher that preserves complete timestamp information"""
    
    def __init__(self, time_window_seconds=1, angle_tolerance_degrees=0.01, min_elevation=10.0, max_elevation=90.0):
        """
        Initialize matcher
        
        Args:
            time_window_seconds: Maximum time difference allowed (default: 1s)
            angle_tolerance_degrees: Tolerance for azimuth/elevation matching (default: 0.01°)
            min_elevation: Minimum elevation threshold (default: 10.0°)
            max_elevation: Maximum elevation threshold (default: 90.0°)
        """
        self.time_window_seconds = time_window_seconds
        self.angle_tolerance = angle_tolerance_degrees
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        self.forest_data = None
        self.roof_data = None
        
    def load_data(self):
        """Load forest and roof data from pickle files"""
        logger.info("Loading datasets...")
        
        # Try to load roof data
        roof_files = ['OpenSky_GNSS_CubicSpline_Dataset_20250626_172809.pkl']
        self.roof_data = self._load_pickle_data(roof_files, "roof")
        
        # Try to load forest data
        forest_files = ['Forest_GNSS_Comprehensive_Dataset_20250626_180320.pkl']
        self.forest_data = self._load_pickle_data(forest_files, "forest")
        
        if self.forest_data is None:
            raise FileNotFoundError("Could not load forest data from any expected file")
        if self.roof_data is None:
            raise FileNotFoundError("Could not load roof data from any expected file")
        
        logger.info(f"Loaded forest data: {len(self.forest_data):,} observations")
        logger.info(f"Loaded roof data: {len(self.roof_data):,} observations")
        
        # Debug: Show data structure
        self._debug_data_structure()
        
    def _load_pickle_data(self, file_list, data_type):
        """Helper to load pickle data with multiple fallback strategies"""
        for file_path in file_list:
            if Path(file_path).exists():
                logger.info(f"Loading {data_type} data from: {file_path}")
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Try different extraction strategies
                    df = self._extract_dataframe(data, data_type)
                    if df is not None:
                        return df
                        
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        return None
    
    def _extract_dataframe(self, data, data_type):
        """Extract DataFrame from various pickle structures"""
        if isinstance(data, pd.DataFrame):
            return data
        
        if isinstance(data, dict):
            # Try common key names
            key_candidates = [
                f'{data_type}_data',
                'data',
                'df',
                'dataframe',
                'observations'
            ]
            
            for key in key_candidates:
                if key in data and isinstance(data[key], pd.DataFrame):
                    logger.info(f"Found DataFrame under key: {key}")
                    return data[key]
            
            # If no key matches, try first DataFrame found
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    logger.info(f"Using DataFrame found under key: {key}")
                    return value
        
        return None
    
    def _debug_data_structure(self):
        """Debug data structure and show sample data"""
        logger.info("\n" + "="*50)
        logger.info("DATA STRUCTURE DEBUG")
        logger.info("="*50)
        
        # Show columns
        logger.info(f"Forest columns: {list(self.forest_data.columns)}")
        logger.info(f"Roof columns: {list(self.roof_data.columns)}")
        
        # Show sample data
        logger.info(f"\nForest sample data:")
        logger.info(self.forest_data.head(3))
        
        logger.info(f"\nRoof sample data:")
        logger.info(self.roof_data.head(3))
        
    def prepare_data(self):
        """Clean and prepare data with enhanced datetime handling"""
        logger.info("Preparing data...")
        
        # Store original counts
        original_forest_count = len(self.forest_data)
        original_roof_count = len(self.roof_data)
        
        # Handle different datetime column formats
        self._standardize_datetime_columns()
        
        # Required columns
        required_cols = ['datetime', 'satellite_id']
        
        for col in required_cols:
            if col not in self.forest_data.columns:
                raise ValueError(f"Missing column '{col}' in forest data")
            if col not in self.roof_data.columns:
                raise ValueError(f"Missing column '{col}' in roof data")
        
        # Convert datetime
        self.forest_data['datetime'] = pd.to_datetime(self.forest_data['datetime'])
        self.roof_data['datetime'] = pd.to_datetime(self.roof_data['datetime'])
        
        # Add enhanced datetime components
        self._add_datetime_components()
        
        # Standardize satellite IDs (convert to string for consistent comparison)
        self.forest_data['satellite_id'] = self.forest_data['satellite_id'].astype(str)
        self.roof_data['satellite_id'] = self.roof_data['satellite_id'].astype(str)
        
        # Remove SAT satellites if present
        forest_before_sat = len(self.forest_data)
        roof_before_sat = len(self.roof_data)
        
        self.forest_data = self.forest_data[
            ~self.forest_data['satellite_id'].str.startswith('SAT')
        ].copy()
        self.roof_data = self.roof_data[
            ~self.roof_data['satellite_id'].str.startswith('SAT')
        ].copy()
        
        sat_removed_forest = forest_before_sat - len(self.forest_data)
        sat_removed_roof = roof_before_sat - len(self.roof_data)
        
        if sat_removed_forest > 0 or sat_removed_roof > 0:
            logger.info(f"Removed SAT satellites: Forest {sat_removed_forest:,}, Roof {sat_removed_roof:,}")
        
        # Check for elevation and azimuth columns
        self.has_geometry = (
            'elevation' in self.forest_data.columns and 'azimuth' in self.forest_data.columns and
            'elevation' in self.roof_data.columns and 'azimuth' in self.roof_data.columns
        )
        
        if self.has_geometry:
            logger.info("✅ Found elevation and azimuth columns - applying elevation filter")
            
            # Apply elevation filter
            forest_before_elev = len(self.forest_data)
            roof_before_elev = len(self.roof_data)
            
            self.forest_data = self.forest_data[
                (self.forest_data['elevation'] >= self.min_elevation) & 
                (self.forest_data['elevation'] <= self.max_elevation)
            ].copy()
            
            self.roof_data = self.roof_data[
                (self.roof_data['elevation'] >= self.min_elevation) & 
                (self.roof_data['elevation'] <= self.max_elevation)
            ].copy()
            
            elev_removed_forest = forest_before_elev - len(self.forest_data)
            elev_removed_roof = roof_before_elev - len(self.roof_data)
            
            logger.info(f"Elevation filter removed: Forest {elev_removed_forest:,}, Roof {elev_removed_roof:,}")
        
        # Convert datetime to timestamp for faster comparison
        self.forest_data['timestamp'] = self.forest_data['datetime'].astype('int64') // 10**9
        self.roof_data['timestamp'] = self.roof_data['datetime'].astype('int64') // 10**9
        
        logger.info(f"Final prepared data:")
        logger.info(f"  Forest: {len(self.forest_data):,} observations")
        logger.info(f"  Roof: {len(self.roof_data):,} observations")
        
    def _standardize_datetime_columns(self):
        """Standardize datetime columns from different formats"""
        # For forest data - if 'datetime' column doesn't exist, try to create it
        if 'datetime' not in self.forest_data.columns:
            if 'day_of_year' in self.forest_data.columns and 'seconds_of_day' in self.forest_data.columns:
                logger.info("Creating datetime from day_of_year and seconds_of_day for forest data")
                # Assuming year 2020 - you may need to adjust this
                year = 2020
                self.forest_data['datetime'] = pd.to_datetime(f'{year}-01-01') + \
                    pd.to_timedelta(self.forest_data['day_of_year'] - 1, unit='D') + \
                    pd.to_timedelta(self.forest_data['seconds_of_day'], unit='s')
        
        # For roof data - similar logic
        if 'datetime' not in self.roof_data.columns:
            if 'day_of_year' in self.roof_data.columns and 'corrected_seconds' in self.roof_data.columns:
                logger.info("Creating datetime from day_of_year and corrected_seconds for roof data")
                year = 2020
                self.roof_data['datetime'] = pd.to_datetime(f'{year}-01-01') + \
                    pd.to_timedelta(self.roof_data['day_of_year'] - 1, unit='D') + \
                    pd.to_timedelta(self.roof_data['corrected_seconds'], unit='s')
            elif 'day_of_year' in self.roof_data.columns and 'hour' in self.roof_data.columns:
                logger.info("Creating datetime from day_of_year, hour, minute, second for roof data")
                year = 2020
                base_date = pd.to_datetime(f'{year}-01-01') + \
                    pd.to_timedelta(self.roof_data['day_of_year'] - 1, unit='D')
                
                time_delta = pd.to_timedelta(self.roof_data['hour'], unit='h') + \
                            pd.to_timedelta(self.roof_data['minute'], unit='m') + \
                            pd.to_timedelta(self.roof_data['second'], unit='s')
                
                self.roof_data['datetime'] = base_date + time_delta
    
    def _add_datetime_components(self):
        """Add comprehensive datetime components to both datasets"""
        for data, name in [(self.forest_data, 'forest'), (self.roof_data, 'roof')]:
            logger.info(f"Adding datetime components to {name} data")
            
            # Extract datetime components
            data['year'] = data['datetime'].dt.year
            data['month'] = data['datetime'].dt.month
            data['day'] = data['datetime'].dt.day
            data['hour'] = data['datetime'].dt.hour
            data['minute'] = data['datetime'].dt.minute
            data['second'] = data['datetime'].dt.second
            data['microsecond'] = data['datetime'].dt.microsecond
            
            # Calculate day of year if not present
            if 'day_of_year' not in data.columns:
                data['day_of_year'] = data['datetime'].dt.dayofyear
            
            # Calculate seconds of day if not present
            if 'seconds_of_day' not in data.columns:
                data['seconds_of_day'] = (
                    data['hour'] * 3600 + 
                    data['minute'] * 60 + 
                    data['second'] + 
                    data['microsecond'] / 1e6
                )
    
    def match_observations(self):
        """Match forest and roof observations with complete timestamp preservation"""
        logger.info(f"\n🔗 STARTING OBSERVATION MATCHING")
        logger.info("-" * 50)
        
        start_time = datetime.now()
        matches = []
        
        # Get common satellites
        forest_sats = set(self.forest_data['satellite_id'].unique())
        roof_sats = set(self.roof_data['satellite_id'].unique())
        common_sats = forest_sats.intersection(roof_sats)
        
        logger.info(f"Common satellites ({len(common_sats)}): {sorted(common_sats)}")
        
        if len(common_sats) == 0:
            logger.error("❌ No common satellites found!")
            return pd.DataFrame()
        
        # Process each satellite
        for sat_id in sorted(common_sats):
            logger.info(f"Processing satellite {sat_id}...")
            
            # Filter data for this satellite
            forest_sat = self.forest_data[self.forest_data['satellite_id'] == sat_id].copy()
            roof_sat = self.roof_data[self.roof_data['satellite_id'] == sat_id].copy()
            
            if len(forest_sat) == 0 or len(roof_sat) == 0:
                continue
            
            # Sort by timestamp for efficient searching
            forest_sat = forest_sat.sort_values('timestamp')
            roof_sat = roof_sat.sort_values('timestamp')
            
            # Find matches for this satellite
            sat_matches = self._find_matches_for_satellite_enhanced(forest_sat, roof_sat, sat_id)
            
            if len(sat_matches) > 0:
                matches.extend(sat_matches)
                logger.info(f"  ✅ Found {len(sat_matches):,} matches")
            else:
                logger.info(f"  ❌ No matches found")
        
        if not matches:
            logger.warning("⚠️ No matches found for any satellite")
            return pd.DataFrame()
        
        # Convert to DataFrame with enhanced output
        result_df = pd.DataFrame(matches)
        
        # Add SNR data if available
        result_df = self._add_snr_data(result_df)
        
        elapsed = datetime.now() - start_time
        logger.info(f"✅ Matching complete: {len(result_df):,} matches in {elapsed.total_seconds():.1f}s")
        
        return result_df
    
    def _find_matches_for_satellite_enhanced(self, forest_sat, roof_sat, sat_id):
        """Find matches with complete timestamp information preserved"""
        matches = []
        
        # Convert to numpy arrays for faster access
        roof_times = roof_sat['timestamp'].values
        roof_indices = roof_sat.index.values
        
        if self.has_geometry:
            roof_elevations = roof_sat['elevation'].values
            roof_azimuths = roof_sat['azimuth'].values
        
        # For each forest observation, find matching roof observations
        for forest_idx, forest_row in forest_sat.iterrows():
            forest_time = forest_row['timestamp']
            
            # Step 1: Find roof observations within time window
            time_mask = np.abs(roof_times - forest_time) <= self.time_window_seconds
            time_candidates = roof_indices[time_mask]
            
            if len(time_candidates) == 0:
                continue
            
            # Step 2: If we have geometry data, filter by azimuth and elevation
            if self.has_geometry:
                forest_elevation = forest_row['elevation']
                forest_azimuth = forest_row['azimuth']
                
                geometry_matches = []
                for roof_idx in time_candidates:
                    roof_idx_pos = np.where(roof_indices == roof_idx)[0][0]
                    roof_elevation = roof_elevations[roof_idx_pos]
                    roof_azimuth = roof_azimuths[roof_idx_pos]
                    
                    # Check if geometry matches within tolerance
                    elev_diff = abs(forest_elevation - roof_elevation)
                    azim_diff = abs(forest_azimuth - roof_azimuth)
                    
                    # Handle azimuth wrap-around (0°/360°)
                    if azim_diff > 180:
                        azim_diff = 360 - azim_diff
                    
                    if elev_diff <= self.angle_tolerance and azim_diff <= self.angle_tolerance:
                        geometry_matches.append(roof_idx)
                
                candidates = geometry_matches
            else:
                candidates = time_candidates.tolist()
            
            # Step 3: From remaining candidates, pick the one closest in time
            if len(candidates) > 0:
                best_roof_idx = None
                best_time_diff = float('inf')
                
                for roof_idx in candidates:
                    roof_time = self.roof_data.loc[roof_idx, 'timestamp']
                    time_diff = abs(forest_time - roof_time)
                    
                    if time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_roof_idx = roof_idx
                
                if best_roof_idx is not None:
                    roof_row = self.roof_data.loc[best_roof_idx]
                    
                    # Create comprehensive match record with all timestamp information
                    match = {
                        # Basic matching info
                        'satellite_id': sat_id,
                        'forest_idx': forest_idx,
                        'roof_idx': best_roof_idx,
                        'time_diff_seconds': best_time_diff,
                        
                        # Forest datetime information
                        'forest_datetime': forest_row['datetime'],
                        'forest_year': forest_row['year'],
                        'forest_month': forest_row['month'],
                        'forest_day': forest_row['day'],
                        'forest_day_of_year': forest_row['day_of_year'],
                        'forest_hour': forest_row['hour'],
                        'forest_minute': forest_row['minute'],
                        'forest_second': forest_row['second'],
                        'forest_seconds_of_day': forest_row['seconds_of_day'],
                        
                        # Roof datetime information
                        'roof_datetime': roof_row['datetime'],
                        'roof_year': roof_row['year'],
                        'roof_month': roof_row['month'],
                        'roof_day': roof_row['day'],
                        'roof_day_of_year': roof_row['day_of_year'],
                        'roof_hour': roof_row['hour'],
                        'roof_minute': roof_row['minute'],
                        'roof_second': roof_row['second'],
                        'roof_seconds_of_day': roof_row['seconds_of_day'],
                    }
                    
                    # Add geometry if available
                    if self.has_geometry:
                        match.update({
                            'forest_elevation': forest_row['elevation'],
                            'forest_azimuth': forest_row['azimuth'],
                            'roof_elevation': roof_row['elevation'],
                            'roof_azimuth': roof_row['azimuth'],
                            'elevation_diff': abs(forest_row['elevation'] - roof_row['elevation']),
                            'azimuth_diff': min(
                                abs(forest_row['azimuth'] - roof_row['azimuth']),
                                360 - abs(forest_row['azimuth'] - roof_row['azimuth'])
                            )
                        })
                    
                    matches.append(match)
        
        return matches
    
    def _add_snr_data(self, matches_df):
        """Add SNR data and calculate delta SNR if available"""
        if len(matches_df) == 0:
            return matches_df
        
        # Find SNR columns
        forest_snr_cols = [col for col in self.forest_data.columns 
                          if 'snr' in col.lower() or 'c/n' in col.lower()]
        roof_snr_cols = [col for col in self.roof_data.columns 
                        if 'snr' in col.lower() or 'c/n' in col.lower()]
        
        if forest_snr_cols and roof_snr_cols:
            logger.info(f"Adding SNR data")
            
            # Get SNR values for matched observations
            forest_snr_values = self.forest_data.loc[matches_df['forest_idx'], forest_snr_cols[0]]
            roof_snr_values = self.roof_data.loc[matches_df['roof_idx'], roof_snr_cols[0]]
            
            matches_df = matches_df.copy()
            matches_df['forest_snr'] = forest_snr_values.values
            matches_df['roof_snr'] = roof_snr_values.values
            matches_df['delta_snr'] = matches_df['forest_snr'] - matches_df['roof_snr']
        
        return matches_df
    
    def run_matching(self, save_results=True):
        """Run the complete matching process"""
        logger.info("="*60)
        logger.info("ENHANCED GNSS MATCHER WITH COMPLETE TIMESTAMP OUTPUT")
        logger.info(f"Time window: {self.time_window_seconds} seconds")
        logger.info(f"Angle tolerance: {self.angle_tolerance} degrees")
        logger.info(f"Elevation range: {self.min_elevation}° - {self.max_elevation}°")
        logger.info("="*60)
        
        try:
            # Load and prepare data
            self.load_data()
            self.prepare_data()
            
            # Check if we have data after filtering
            if len(self.forest_data) == 0:
                logger.error("❌ No forest data remaining after filtering!")
                return None
            if len(self.roof_data) == 0:
                logger.error("❌ No roof data remaining after filtering!")
                return None
            
            # Run matching
            matches = self.match_observations()
            
            if len(matches) > 0:
                # Show statistics
                self._show_statistics(matches)
                
                # Save results
                if save_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'gnss_matches_enhanced_{timestamp}.csv'
                    matches.to_csv(filename, index=False)
                    logger.info(f"💾 Results saved to: {filename}")
                
                return matches
            else:
                logger.error("❌ No matches found!")
                return None
                
        except Exception as e:
            logger.error(f"Matching failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _show_statistics(self, matches_df):
        """Show comprehensive matching statistics"""
        logger.info("\n" + "="*50)
        logger.info("ENHANCED MATCHING RESULTS")
        logger.info("="*50)
        logger.info(f"Total matches found: {len(matches_df):,}")
        
        # Time difference statistics
        time_stats = matches_df['time_diff_seconds'].describe()
        logger.info(f"\nTime differences (seconds):")
        logger.info(f"  Mean: {time_stats['mean']:.3f}s")
        logger.info(f"  Median: {time_stats['50%']:.3f}s")
        logger.info(f"  Max: {time_stats['max']:.3f}s")
        
        # Show sample of enhanced output
        logger.info(f"\nSample match with complete timestamp info:")
        if len(matches_df) > 0:
            sample = matches_df.iloc[0]
            logger.info(f"  Satellite: {sample['satellite_id']}")
            logger.info(f"  Forest: DOY {sample['forest_day_of_year']}, {sample['forest_hour']:02d}:{sample['forest_minute']:02d}:{sample['forest_second']:02d}")
            logger.info(f"  Roof:   DOY {sample['roof_day_of_year']}, {sample['roof_hour']:02d}:{sample['roof_minute']:02d}:{sample['roof_second']:02d}")
            logger.info(f"  Time diff: {sample['time_diff_seconds']:.3f}s")


def main():
    """Run the enhanced GNSS matcher"""
    # Create matcher
    matcher = EnhancedGNSSMatcher(
        time_window_seconds=5,
        angle_tolerance_degrees=5,
        min_elevation=10.0,
        max_elevation=90.0
    )
    
    # Run matching
    results = matcher.run_matching()
    
    if results is not None:
        print(f"\n✅ SUCCESS! Found {len(results):,} matches")
        print(f"📊 Complete timestamp information preserved in output")
        
        # Show sample columns
        print(f"\nOutput columns include:")
        print("- Complete forest timestamp info (day_of_year, hour, minute, second, seconds_of_day)")
        print("- Complete roof timestamp info (day_of_year, hour, minute, second, seconds_of_day)")
        print("- Satellite ID and geometry information")
        print("- SNR values and delta SNR")
        
        return results
    else:
        print("\n❌ No matches found")
        return None


if __name__ == "__main__":
    main()