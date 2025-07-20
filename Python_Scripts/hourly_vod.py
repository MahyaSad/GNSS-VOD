#!/usr/bin/env python3
"""
ENHANCED VOD ANALYSIS WITH IMPROVED ERROR HANDLING
================================================
Key improvements:
1. Better error handling for missing columns
2. More flexible timestamp detection
3. Improved spatial averaging efficiency
4. Better memory management for large datasets
5. Enhanced validation steps

Author: Enhanced Analysis Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
from datetime import datetime, timedelta
import glob
from scipy.spatial.distance import cdist
from numba import jit
from sklearn.neighbors import BallTree
warnings.filterwarnings('ignore')

class EnhancedVODAnalyzer:
    """Enhanced VOD analysis with improved error handling and efficiency"""
    
    def __init__(self, min_elevation=10.0, neighbor_radius=0.5, min_transmissivity=0.1, max_transmissivity=2.0):
        """
        Initialize enhanced analyzer
        
        Args:
            min_elevation: Minimum elevation threshold (default: 10°)
            neighbor_radius: Radius for spatial averaging (default: 0.5°)
            min_transmissivity: Minimum transmissivity threshold (default: 0.1)
            max_transmissivity: Maximum transmissivity threshold (default: 2.0)
        """
        self.min_elevation = min_elevation
        self.neighbor_radius = neighbor_radius
        self.min_transmissivity = min_transmissivity
        self.max_transmissivity = max_transmissivity
        self.data = None
        self.filtered_data = None
        
        print(f"🚀 ENHANCED VOD ANALYZER INITIALIZED")
        print(f"   Min elevation: {min_elevation}°")
        print(f"   Neighbor radius: {neighbor_radius}°")
        print(f"   Transmissivity range: {min_transmissivity} - {max_transmissivity}")
    
    def validate_input_data(self, df):
        """Validate that the input data has the required columns"""
        print(f"\n🔍 VALIDATING INPUT DATA")
        print("-" * 40)
        
        # Check for exact column names from your CSV structure
        required_columns = [
            'satellite_id',
            'forest_snr', 
            'roof_snr',
            'forest_elevation',
            'forest_azimuth', 
            'forest_datetime',
            'forest_day_of_year',
            'forest_hour',
            'forest_minute', 
            'forest_second',
            'forest_seconds_of_day',
            'delta_snr'  # Already calculated in your CSV
        ]
        
        print(f"   📋 Available columns: {list(df.columns)}")
        
        # Check which required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        present_columns = [col for col in required_columns if col in df.columns]
        
        print(f"   ✅ Present required columns: {present_columns}")
        
        if missing_columns:
            print(f"   ⚠️ Missing columns: {missing_columns}")
            # Check if we can work without them
            critical_missing = [col for col in missing_columns if col in [
                'satellite_id', 'forest_elevation', 'forest_azimuth', 'forest_datetime'
            ]]
            if critical_missing:
                raise ValueError(f"❌ Missing critical columns: {critical_missing}")
        
        # If delta_snr is missing, we can calculate it
        if 'delta_snr' not in df.columns:
            if 'forest_snr' in df.columns and 'roof_snr' in df.columns:
                print(f"   🔧 Calculating delta_snr = forest_snr - roof_snr")
                df['delta_snr'] = df['forest_snr'] - df['roof_snr']
            else:
                raise ValueError("❌ Cannot calculate delta_snr: missing forest_snr or roof_snr")
        
        print(f"   ✅ Input validation successful")
        return df, {}
    
    def load_matched_data(self):
        """Load matched GNSS data with enhanced flexibility"""
        print(f"\n📂 LOADING MATCHED GNSS DATA")
        print("-" * 40)
        
        # Look for matched data files with broader patterns
        csv_patterns = [
            'gnss_matches_enhanced_20250628_185836.csv',
            'gnss_matches_elev*.csv', 
            'gnss_matches_*.csv',
            'matched_data_*.csv',
            '*matched*.csv',
            '*gnss*.csv'
        ]
        
        found_file = None
        for pattern in csv_patterns:
            files = glob.glob(pattern)
            if files:
                # Get the most recent file
                found_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                break
        
        if found_file is None:
            raise FileNotFoundError("❌ No matched GNSS data files found!")
        
        print(f"   📄 Loading: {found_file}")
        
        try:
            # Load data
            raw_data = pd.read_csv(found_file)
            print(f"   📊 Raw data loaded: {len(raw_data):,} observations")
            
            # Validate columns (no need to rename since they match exactly)
            self.data, _ = self.validate_input_data(raw_data)
            
            # Extract timestamp information directly from existing columns
            self._extract_timestamp_info_enhanced()
            
            # Show basic statistics
            print(f"   📊 Delta SNR range: {self.data['delta_snr'].min():.1f} to {self.data['delta_snr'].max():.1f} dB")
            print(f"   📊 Forest SNR range: {self.data['forest_snr'].min():.1f} to {self.data['forest_snr'].max():.1f} dB")
            print(f"   📊 Roof SNR range: {self.data['roof_snr'].min():.1f} to {self.data['roof_snr'].max():.1f} dB")
            print(f"   📊 Elevation range: {self.data['forest_elevation'].min():.1f}° to {self.data['forest_elevation'].max():.1f}°")
            
            # Check for missing values in critical columns
            critical_cols = ['delta_snr', 'forest_elevation', 'forest_azimuth', 'forest_datetime']
            missing_counts = self.data[critical_cols].isnull().sum()
            if missing_counts.any():
                print(f"   ⚠️ Missing values found:")
                for col, count in missing_counts.items():
                    if count > 0:
                        print(f"      {col}: {count} missing")
                
                # Remove rows with missing critical values
                before_count = len(self.data)
                self.data = self.data.dropna(subset=critical_cols)
                after_count = len(self.data)
                if before_count != after_count:
                    print(f"   🗑️ Removed {before_count - after_count} rows with missing values")
            
            # Show satellite distribution
            sat_counts = self.data['satellite_id'].value_counts()
            print(f"   🛰️ Satellites: {len(sat_counts)} ({', '.join(sat_counts.head(5).index.astype(str).tolist())}...)")
            
        except Exception as e:
            raise Exception(f"❌ Error loading data: {e}")
    
    def _extract_timestamp_info_enhanced(self):
        """Enhanced timestamp extraction using existing forest datetime columns"""
        print(f"\n🕒 EXTRACTING TIMESTAMP INFORMATION")
        print("-" * 40)
        
        # Use the existing forest_datetime column
        if 'forest_datetime' in self.data.columns:
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['forest_datetime'])
                print(f"   ✅ Using forest_datetime column")
            except:
                print(f"   ⚠️ Failed to parse forest_datetime, reconstructing from components")
                self._reconstruct_datetime_from_components()
        else:
            print(f"   ⚠️ No forest_datetime column found, reconstructing from components")
            self._reconstruct_datetime_from_components()
        
        # Use existing forest timing columns directly
        if 'forest_day_of_year' in self.data.columns:
            self.data['doy'] = self.data['forest_day_of_year']
            print(f"   ✅ Using existing forest_day_of_year")
        else:
            self.data['doy'] = self.data['timestamp'].dt.dayofyear
            print(f"   🔧 Calculated DOY from timestamp")
        
        # Use existing forest time components
        if 'forest_hour' in self.data.columns:
            self.data['hour'] = self.data['forest_hour']
            print(f"   ✅ Using existing forest_hour")
        else:
            self.data['hour'] = self.data['timestamp'].dt.hour
        
        if 'forest_minute' in self.data.columns:
            self.data['minute'] = self.data['forest_minute']
            print(f"   ✅ Using existing forest_minute")
        else:
            self.data['minute'] = self.data['timestamp'].dt.minute
        
        if 'forest_second' in self.data.columns:
            self.data['second'] = self.data['forest_second']
            print(f"   ✅ Using existing forest_second")
        else:
            self.data['second'] = self.data['timestamp'].dt.second
        
        # Use forest_seconds_of_day if available (useful for hourly aggregation)
        if 'forest_seconds_of_day' in self.data.columns:
            self.data['seconds_of_day'] = self.data['forest_seconds_of_day']
            print(f"   ✅ Using existing forest_seconds_of_day")
        else:
            self.data['seconds_of_day'] = (
                self.data['hour'] * 3600 + 
                self.data['minute'] * 60 + 
                self.data['second']
            )
            print(f"   🔧 Calculated seconds_of_day from time components")
        
        print(f"   📅 Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"   📅 DOY range: {self.data['doy'].min()} to {self.data['doy'].max()}")
        
        # Calculate time span
        time_span = self.data['timestamp'].max() - self.data['timestamp'].min()
        print(f"   📊 Total time span: {time_span}")
    
    def _reconstruct_datetime_from_components(self):
        """Reconstruct datetime from individual year, month, day, hour, minute, second components"""
        try:
            self.data['timestamp'] = pd.to_datetime({
                'year': self.data['forest_year'],
                'month': self.data['forest_month'], 
                'day': self.data['forest_day'],
                'hour': self.data['forest_hour'],
                'minute': self.data['forest_minute'],
                'second': self.data['forest_second']
            })
            print(f"   🔧 Reconstructed timestamp from individual components")
        except Exception as e:
            print(f"   ❌ Failed to reconstruct datetime: {e}")
            # Fallback to synthetic timestamps
            self._create_synthetic_timestamps()
    
    def _create_synthetic_timestamps(self):
        """Create synthetic timestamps based on data order"""
        base_time = datetime(2024, 1, 1)  # Start from beginning of 2024
        # Assume observations are spread over 8 months (240 days)
        time_intervals = pd.timedelta_range(start='0 hours', periods=len(self.data), freq='1H')
        self.data['timestamp'] = base_time + time_intervals
        print(f"   🔧 Created synthetic timestamps over 8 months")
    
    def calculate_spatial_averages_optimized(self, data_subset, column_name):
        """Calculate spatial averages using optimized BallTree approach"""
        print(f"\n🎯 CALCULATING SPATIAL AVERAGES FOR {column_name.upper()} (OPTIMIZED)")
        print("-" * 40)
        
        # Create coordinate matrix for BallTree (lat, lon in radians)
        coords = np.column_stack([
            np.radians(data_subset['forest_elevation']),  # Using elevation as proxy for latitude
            np.radians(data_subset['forest_azimuth'])     # Using azimuth as proxy for longitude
        ])
        
        print(f"   📊 Building BallTree for {len(coords):,} points...")
        
        # Build BallTree with haversine metric for spherical distances
        tree = BallTree(coords, metric='haversine')
        
        # Convert neighbor radius from degrees to radians
        radius_rad = np.radians(self.neighbor_radius)
        
        # Query for neighbors within radius
        print(f"   🔍 Finding neighbors within {self.neighbor_radius}° radius...")
        neighbor_indices, neighbor_distances = tree.query_radius(
            coords, r=radius_rad, return_distance=True
        )
        
        # Calculate local means
        print(f"   📊 Calculating local spatial means...")
        local_means = []
        neighbor_counts = []
        
        values = data_subset[column_name].values
        
        for i, (indices, distances) in enumerate(zip(neighbor_indices, neighbor_distances)):
            if len(indices) > 1:  # At least one neighbor besides itself
                # Get values for neighbors
                neighbor_values = values[indices]
                
                # Remove NaN values
                valid_values = neighbor_values[~np.isnan(neighbor_values)]
                
                if len(valid_values) > 0:
                    local_means.append(np.mean(valid_values))
                    neighbor_counts.append(len(valid_values))
                else:
                    local_means.append(np.nan)
                    neighbor_counts.append(0)
            else:
                local_means.append(np.nan)
                neighbor_counts.append(0)
        
        # Add local means to data
        data_subset = data_subset.copy()
        data_subset[f'{column_name}_local_mean'] = local_means
        data_subset[f'{column_name}_neighbor_count'] = neighbor_counts
        
        # Calculate global mean from valid local means
        valid_means = np.array(local_means)[~np.isnan(local_means)]
        global_mean = np.mean(valid_means) if len(valid_means) > 0 else np.nan
        
        print(f"   📊 Results:")
        print(f"      Local mean range: {np.min(valid_means):.4f} to {np.max(valid_means):.4f}")
        print(f"      Global mean: {global_mean:.4f}")
        print(f"      Average neighbors per point: {np.mean(neighbor_counts[np.array(neighbor_counts) > 0]):.1f}")
        print(f"      Points with neighbors: {(np.array(neighbor_counts) > 0).sum():,}")
        
        return data_subset, global_mean
    
    def apply_elevation_filter(self):
        """Apply elevation filtering with validation"""
        print(f"\n🔧 APPLYING ELEVATION FILTER")
        print("-" * 40)
        
        original_count = len(self.data)
        print(f"   📊 Original observations: {original_count:,}")
        
        # Validate elevation values
        invalid_elevations = (
            (self.data['forest_elevation'] < 0) | 
            (self.data['forest_elevation'] > 90) |
            (self.data['forest_elevation'].isna())
        ).sum()
        
        if invalid_elevations > 0:
            print(f"   ⚠️ Found {invalid_elevations} invalid elevation values")
            self.data = self.data[
                (self.data['forest_elevation'] >= 0) & 
                (self.data['forest_elevation'] <= 90) &
                (~self.data['forest_elevation'].isna())
            ].copy()
        
        # Apply elevation filter
        elevation_mask = self.data['forest_elevation'] >= self.min_elevation
        self.data = self.data[elevation_mask].copy()
        
        filtered_count = len(self.data)
        removed_count = original_count - filtered_count
        
        print(f"   📊 After elevation filter (≥{self.min_elevation}°): {filtered_count:,}")
        print(f"   📊 Removed: {removed_count:,} ({100*removed_count/original_count:.1f}%)")
        
        if filtered_count == 0:
            raise ValueError("❌ No data remaining after elevation filter!")
        
        print(f"   📊 New elevation range: {self.data['forest_elevation'].min():.1f}° to {self.data['forest_elevation'].max():.1f}°")
    
    def apply_spatial_domain_filter(self):
        """Apply spatial domain filtering using azimuth and elevation with NE/NW artifact removal"""
        print(f"\n🌐 APPLYING SPATIAL DOMAIN FILTER WITH ARTIFACT REMOVAL")
        print("-" * 40)
        
        # Check for valid azimuth and elevation ranges
        az_min, az_max = self.data['forest_azimuth'].min(), self.data['forest_azimuth'].max()
        el_min, el_max = self.data['forest_elevation'].min(), self.data['forest_elevation'].max()
        
        print(f"   📊 Azimuth range: {az_min:.1f}° to {az_max:.1f}°")
        print(f"   📊 Elevation range: {el_min:.1f}° to {el_max:.1f}°")
        
        # Convert negative azimuths to 0-360 range
        if az_min < 0:
            print(f"   🔧 Converting negative azimuths to 0-360° range")
            self.data['forest_azimuth'] = self.data['forest_azimuth'] % 360
            if 'roof_azimuth' in self.data.columns:
                self.data['roof_azimuth'] = self.data['roof_azimuth'] % 360
        
        original_count = len(self.data)
        
        # Basic spatial domain filter
        valid_mask = (
            (self.data['forest_elevation'] >= self.min_elevation) & 
            (self.data['forest_elevation'] <= 90) &
            (self.data['forest_azimuth'] >= 0) & 
            (self.data['forest_azimuth'] < 360)
        )
        
        self.data = self.data[valid_mask].copy()
        basic_filtered_count = len(self.data)
        
        print(f"   📊 After basic spatial filter: {basic_filtered_count:,}")
        
        # Apply NE and NW artifact masks
        print(f"   🔧 Applying NE/NW artifact removal...")
        
        # Northeast artifacts (0° to 70° azimuth, elevation ≤ 30°)
        mask_NE = ((self.data['forest_azimuth'] >= 0) & 
                   (self.data['forest_azimuth'] <= 70) & 
                   (self.data['forest_elevation'] <= 30))
        
        # Northwest artifacts (280° to 340° azimuth, elevation ≤ 40°)
        mask_NW = ((self.data['forest_azimuth'] >= 280) & 
                   (self.data['forest_azimuth'] <= 340) & 
                   (self.data['forest_elevation'] <= 40))
        
        # Count artifacts
        ne_artifacts = mask_NE.sum()
        nw_artifacts = mask_NW.sum()
        total_artifacts = ne_artifacts + nw_artifacts
        
        print(f"   📊 NE artifacts (0°-70° az, ≤30° el): {ne_artifacts:,}")
        print(f"   📊 NW artifacts (280°-340° az, ≤40° el): {nw_artifacts:,}")
        print(f"   📊 Total artifacts to remove: {total_artifacts:,}")
        
        # Remove artifact observations (keep elevation > 10° requirement)
        artifact_mask = mask_NE | mask_NW
        self.data = self.data[~artifact_mask].copy()
        
        # Final check: ensure elevation is still > min_elevation
        final_elevation_mask = self.data['forest_elevation'] >= self.min_elevation
        self.data = self.data[final_elevation_mask].copy()
        
        filtered_count = len(self.data)
        total_removed = original_count - filtered_count
        
        print(f"   ✅ Spatial domain filter complete: {filtered_count:,} observations retained")
        print(f"   📊 Total removed: {total_removed:,} ({100*total_removed/original_count:.1f}%)")
        print(f"   📊 Final elevation range: {self.data['forest_elevation'].min():.1f}° to {self.data['forest_elevation'].max():.1f}°")
        print(f"   📊 Final azimuth range: {self.data['forest_azimuth'].min():.1f}° to {self.data['forest_azimuth'].max():.1f}°")
    
    def calculate_delta_snr_and_transmissivity(self):
        """Calculate transmissivity from existing delta SNR"""
        print(f"\n🧮 USING EXISTING DELTA SNR AND CALCULATING TRANSMISSIVITY")
        print("-" * 40)
        
        # Use the existing delta_snr from your CSV
        print(f"   ✅ Using existing delta_snr column from input CSV")
        print(f"   📊 Delta SNR range: {self.data['delta_snr'].min():.2f} to {self.data['delta_snr'].max():.2f} dB")
        print(f"   📊 Delta SNR mean: {self.data['delta_snr'].mean():.2f} ± {self.data['delta_snr'].std():.2f} dB")
        
        # Verify delta_snr calculation (optional check)
        calculated_delta = self.data['forest_snr'] - self.data['roof_snr']
        if not np.allclose(self.data['delta_snr'], calculated_delta, rtol=1e-05, atol=1e-08, equal_nan=True):
            print(f"   ⚠️ Warning: Existing delta_snr doesn't match forest_snr - roof_snr calculation")
            print(f"   🔧 Using existing delta_snr values from CSV")
        
        # Calculate transmissivity (Equation 10: T = 10^(ΔS/10))
        self.data['transmissivity'] = 10.0**(self.data['delta_snr'] / 10.0)
        
        print(f"   📊 Transmissivity range: {self.data['transmissivity'].min():.4f} to {self.data['transmissivity'].max():.4f}")
        print(f"   📊 Transmissivity mean: {self.data['transmissivity'].mean():.4f} ± {self.data['transmissivity'].std():.4f}")
        
        # Show transmissivity distribution
        trans_above_1 = (self.data['transmissivity'] > 1).sum()
        print(f"   📊 Transmissivity > 1: {trans_above_1:,} observations ({100*trans_above_1/len(self.data):.1f}%)")
    
    def create_hourly_vod_timeseries(self):
        """Create hourly VOD time series for output CSV"""
        print(f"\n📊 CREATING HOURLY VOD TIME SERIES")
        print("-" * 40)
        
        if self.filtered_data is None:
            raise ValueError("❌ No filtered data available. Run filtering steps first.")
        
        # Create hourly time bins
        self.filtered_data['hour_bin'] = (
            self.filtered_data['timestamp'].dt.floor('H')
        )
        
        # Group by hourly bins and calculate statistics
        hourly_stats = self.filtered_data.groupby('hour_bin').agg({
            'vod_processed': ['mean', 'std', 'count'],
            'transmissivity': ['mean', 'std'],
            'delta_snr': ['mean', 'std'],
            'forest_elevation': ['mean', 'min', 'max'],
            'forest_azimuth': ['mean', 'min', 'max'],
            'doy': 'first',  # DOY should be the same within an hour
            'satellite_id': 'nunique'  # Number of unique satellites
        }).round(4)
        
        # Flatten column names
        hourly_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                               for col in hourly_stats.columns]
        
        # Reset index to make hour_bin a column
        hourly_stats = hourly_stats.reset_index()
        
        # Extract time components for the hourly data
        hourly_stats['year'] = hourly_stats['hour_bin'].dt.year
        hourly_stats['month'] = hourly_stats['hour_bin'].dt.month
        hourly_stats['day'] = hourly_stats['hour_bin'].dt.day
        hourly_stats['hour'] = hourly_stats['hour_bin'].dt.hour
        hourly_stats['doy'] = hourly_stats['hour_bin'].dt.dayofyear
        
        print(f"   📊 Created hourly time series with {len(hourly_stats)} hours")
        print(f"   📅 Time range: {hourly_stats['hour_bin'].min()} to {hourly_stats['hour_bin'].max()}")
        print(f"   📊 Average observations per hour: {hourly_stats['vod_processed_count'].mean():.1f}")
        
        return hourly_stats

    def run_complete_analysis(self):
        """Run the complete enhanced VOD analysis pipeline"""
        print(f"🚀 STARTING ENHANCED VOD ANALYSIS PIPELINE")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Step 1: Load and validate matched data
            self.load_matched_data()
            
            # Step 2: Apply elevation filter
            self.apply_elevation_filter()
            
            # Step 3: Apply spatial domain filter
            self.apply_spatial_domain_filter()
            
            # Step 4: Use existing delta SNR and calculate transmissivity
            self.calculate_delta_snr_and_transmissivity()
            
            # Step 5: Calculate spatial averages for transmissivity (optimized)
            print(f"\n🎯 STEP 5: TRANSMISSIVITY SPATIAL AVERAGING (OPTIMIZED)")
            data_with_trans_means, transmissivity_global_mean_initial = self.calculate_spatial_averages_optimized(
                self.data, 'transmissivity'
            )
            
            # Step 6: Apply transmissivity threshold filter
            print(f"\n🔧 STEP 6: APPLYING TRANSMISSIVITY THRESHOLD FILTER")
            print(f"   📊 Before threshold filter: {len(data_with_trans_means):,} observations")
            
            # Apply threshold filter to local means
            valid_mask = (
                (data_with_trans_means['transmissivity_local_mean'] >= self.min_transmissivity) &
                (data_with_trans_means['transmissivity_local_mean'] <= self.max_transmissivity) &
                (~data_with_trans_means['transmissivity_local_mean'].isna())
            )
            
            self.filtered_data = data_with_trans_means[valid_mask].copy()
            filtered_count = len(self.filtered_data)
            removed_count = len(data_with_trans_means) - filtered_count
            
            print(f"   📊 After transmissivity filter: {filtered_count:,}")
            print(f"   📊 Removed: {removed_count:,} ({100*removed_count/len(data_with_trans_means):.1f}%)")
            
            # Step 7: Calculate VOD and final anomalies
            print(f"\n🧮 STEP 7: CALCULATING VOD AND ANOMALIES")
            
            # Calculate incidence angle from elevation
            self.filtered_data['incidence_angle'] = 90.0 - self.filtered_data['forest_elevation']
            
            # Calculate VOD (Equation 11: VOD = -ln(T) * cos(θ))
            epsilon = 1e-10  # Small value to avoid log(0)
            safe_transmissivity = np.maximum(self.filtered_data['transmissivity'], epsilon)
            
            self.filtered_data['vod_raw'] = (
                -np.log(safe_transmissivity) * 
                np.cos(np.radians(self.filtered_data['incidence_angle']))
            )
            
            print(f"   📊 VOD range: {self.filtered_data['vod_raw'].min():.4f} to {self.filtered_data['vod_raw'].max():.4f}")
            
            # Calculate spatial averages for VOD
            self.filtered_data, vod_global_mean = self.calculate_spatial_averages_optimized(
                self.filtered_data, 'vod_raw'
            )
            
            # Recalculate transmissivity local means for filtered data
            self.filtered_data, transmissivity_global_mean = self.calculate_spatial_averages_optimized(
                self.filtered_data, 'transmissivity'
            )
            
            # Calculate VOD anomalies: individual VOD - local mean + global mean
            self.filtered_data['vod_anomaly'] = (
                self.filtered_data['vod_raw'] - 
                self.filtered_data['vod_raw_local_mean'] + 
                vod_global_mean
            )
            
            # Final processed VOD
            self.filtered_data['vod_processed'] = self.filtered_data['vod_anomaly']
            
            print(f"   📊 VOD anomaly range: {self.filtered_data['vod_anomaly'].min():.4f} to {self.filtered_data['vod_anomaly'].max():.4f}")
            
            # Step 8: Create hourly VOD time series
            hourly_vod = self.create_hourly_vod_timeseries()
            
            # Step 9: Save results
            output_dir = Path("vod_analysis_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save individual observations
            individual_filename = output_dir / f"vod_individual_observations_{timestamp}.csv"
            self.filtered_data.to_csv(individual_filename, index=False)
            print(f"   💾 Saved individual observations: {individual_filename}")
            
            # Save hourly time series
            hourly_filename = output_dir / f"vod_hourly_timeseries_{timestamp}.csv"
            hourly_vod.to_csv(hourly_filename, index=False)
            print(f"   💾 Saved hourly time series: {hourly_filename}")
            
            total_time = time.time() - start_time
            print(f"\n✅ ENHANCED ANALYSIS COMPLETED!")
            print(f"⏱️  Total processing time: {total_time:.1f} seconds")
            print(f"📁 Results saved in: vod_analysis_results/")
            
            return {
                'original_data': self.data,
                'filtered_data': self.filtered_data,
                'hourly_vod': hourly_vod,
                'processing_time': total_time,
                'transmissivity_global_mean': transmissivity_global_mean,
                'vod_global_mean': vod_global_mean,
                'individual_file': individual_filename,
                'hourly_file': hourly_filename
            }
            
        except Exception as e:
            print(f"\n💥 ERROR in enhanced analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run the enhanced VOD analysis"""
    print("🚀 ENHANCED VOD ANALYSIS WITH IMPROVED ERROR HANDLING")
    print("=" * 60)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedVODAnalyzer(
        min_elevation=10.0,          # 10° elevation filter
        neighbor_radius=0.5,         # 0.5° spatial averaging  
        min_transmissivity=0.0001,     # Minimum transmissivity threshold
        max_transmissivity=40        # Maximum transmissivity threshold
    )
    
    # Run enhanced analysis
    results = analyzer.run_complete_analysis()
    
    if results is not None:
        print(f"\n🎯 ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📈 Final dataset: {len(results['filtered_data']):,} observations")
        print(f"⏱️  Processing time: {results['processing_time']:.1f} seconds")
        print(f"📁 Individual observations: {results['individual_file']}")
        print(f"📁 Hourly time series: {results['hourly_file']}")
        
        # Display summary statistics
        if 'filtered_data' in results and len(results['filtered_data']) > 0:
            vod_data = results['filtered_data']['vod_processed']
            print(f"\n📊 VOD SUMMARY STATISTICS:")
            print(f"   Mean VOD: {vod_data.mean():.4f}")
            print(f"   Std VOD: {vod_data.std():.4f}")
            print(f"   Min VOD: {vod_data.min():.4f}")
            print(f"   Max VOD: {vod_data.max():.4f}")
            print(f"   Median VOD: {vod_data.median():.4f}")
        
        return results
    else:
        print(f"\n❌ Enhanced analysis failed. Please check error messages above.")
        return None


if __name__ == "__main__":
    results = main()