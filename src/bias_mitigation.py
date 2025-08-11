import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging

class BiasMitigation:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def mitigate_bias(self, df: pd.DataFrame, slice_metrics: Dict) -> pd.DataFrame:
        """
        Apply bias mitigation techniques based on slice performance
        """
        try:
            # 1. Resampling for imbalanced slices
            df = self._balance_slices(df)
            
            # 2. Feature importance weighting
            df = self._adjust_feature_weights(df)
            
            # 3. Threshold adjustment
            df = self._adjust_prediction_thresholds(df)
            
            self.logger.info("Bias mitigation completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in bias mitigation: {str(e)}")
            return df
    
    def _balance_slices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance data across different market conditions"""
        try:
            # Calculate market conditions
            df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            
            # Create balanced samples
            balanced_dfs = []
            
            # Balance based on volatility
            high_vol = df[df['volatility'] > df['volatility'].median()]
            low_vol = df[df['volatility'] <= df['volatility'].median()]
            min_vol_samples = min(len(high_vol), len(low_vol))
            
            balanced_dfs.append(high_vol.sample(n=min_vol_samples, random_state=42))
            balanced_dfs.append(low_vol.sample(n=min_vol_samples, random_state=42))
            
            # Balance based on volume
            high_vol = df[df['volume_ma'] > df['volume_ma'].median()]
            low_vol = df[df['volume_ma'] <= df['volume_ma'].median()]
            min_vol_samples = min(len(high_vol), len(low_vol))
            
            balanced_dfs.append(high_vol.sample(n=min_vol_samples, random_state=42))
            balanced_dfs.append(low_vol.sample(n=min_vol_samples, random_state=42))
            
            # Combine balanced datasets
            df_balanced = pd.concat(balanced_dfs).drop_duplicates()
            
            self.logger.info(f"Balanced dataset created with {len(df_balanced)} samples")
            return df_balanced
            
        except Exception as e:
            self.logger.error(f"Error in slice balancing: {str(e)}")
            return df
    
    def _adjust_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust feature importance weights based on performance"""
        try:
            # Identify numerical features
            numeric_features = df.select_dtypes(include=[np.number]).columns
            
            # Scale features with different weights based on importance
            weighted_features = {}
            for feature in numeric_features:
                if 'price' in feature.lower():
                    weighted_features[feature] = 1.5  # Higher weight for price-related features
                elif 'volume' in feature.lower():
                    weighted_features[feature] = 1.2  # Medium weight for volume
                elif 'Return' in feature:  # For global market returns
                    weighted_features[feature] = 1.3  # Higher weight for market indicators
                else:
                    weighted_features[feature] = 1.0  # Base weight
                    
            # Apply weights
            for feature, weight in weighted_features.items():
                if feature in df.columns:
                    df[feature] = df[feature] * weight
                    
            self.logger.info("Feature weights adjusted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature weight adjustment: {str(e)}")
            return df
    
    def _adjust_prediction_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust prediction thresholds based on market conditions"""
        try:
            if 'Prediction' not in df.columns:
                return df
                
            # Adjust thresholds based on volatility
            volatility = df['Close'].pct_change().rolling(window=20).std()
            
            # More conservative predictions during high volatility
            high_vol_mask = volatility > volatility.median()
            df.loc[high_vol_mask, 'Prediction'] = np.where(
                df.loc[high_vol_mask, 'Prediction_Probability'] > 0.6,  # Higher threshold
                df.loc[high_vol_mask, 'Prediction'],
                0  # Neutral prediction
            )
            
            # Standard threshold for normal volatility
            df.loc[~high_vol_mask, 'Prediction'] = np.where(
                df.loc[~high_vol_mask, 'Prediction_Probability'] > 0.5,  # Normal threshold
                df.loc[~high_vol_mask, 'Prediction'],
                0
            )
            
            self.logger.info("Prediction thresholds adjusted based on market conditions")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in threshold adjustment: {str(e)}")
            return df 