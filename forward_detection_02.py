import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.mixture import GaussianMixture

from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.fft import fft
import warnings

from typing import List, Tuple, Dict

import feature_engineering_01

warnings.filterwarnings('ignore')

class WaferAnomalyDetector:
    """
    Advanced anomaly detection system for semiconductor wafer production data
    Addresses multi-profile correlations, drift robustness, and real-time processing
    """
    
    def __init__(self, contamination=0.1, pca_components=0.95):
        self.contamination = contamination
        self.pca_components = pca_components
        self.pca = None
        self.detectors = {}
        self.drift_baseline = None
        self.processed_samples = 0
        

    
    def fit_drift_robust_model(self, features_scaled, labels=None):
        """
        Fit multiple anomaly detection models with drift robustness.
        Assumes features are already scaled.
        """
        print("Training anomaly detection models on scaled features...")
        
        ##
        # Statistical methods don't need sklearn fit, we'll handle them separately
        # 9. Statistical Outliers (Z-Score) - computed in detect_anomalies
        # 10. Modified Z-Score - computed in detect_anomalies  
        # 11. Spectral Analysis - computed in detect_anomalies
        # 12. Mahalanobis Distance - computed in detect_anomalies
        
        # Apply PCA for dimensionality reduction
        # Can be a float (0.0-1.0) for variance explained or an int for number of components
        self.pca = PCA(n_components=self.pca_components)
        features_pca = self.pca.fit_transform(features_scaled)
        if isinstance(self.pca_components, float) and 0 < self.pca_components < 1.0:
            print(f"PCA selected {self.pca.n_components_} components to explain {self.pca_components:.0%} of variance.")
        else:
            print(f"PCA reduced dimensions to: {features_pca.shape[1]}")
        
        # Initialize the selected detectors
        
        # 1. Isolation Forest
        self.detectors['isolation_forest'] = IsolationForest(
            contamination=self.contamination, 
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Local Outlier Factor
        self.detectors['lof'] = LocalOutlierFactor(
            n_neighbors=min(20, len(features_pca)//3), 
            contamination=self.contamination,
            novelty=False  # novelty=False to use fit_predict
        )
        
        # 3. k-Nearest Neighbors Distance
        k = min(10, len(features_pca)//5)
        self.detectors['knn_distance'] = NearestNeighbors(
            n_neighbors=k+1  # +1 because first neighbor is the point itself
        )
        
        # 4. Mahalanobis Distance is calculated later, no model to fit here.

        # 5. Elliptic Envelope
        n_components = min(5, max(2, len(features_pca)//10))
        self.detectors['gaussian_mixture'] = GaussianMixture(
            n_components=n_components, 
            random_state=42
        )

        # Fit detectors on PCA features (exclude LOF as it's fit during prediction)
        for name, detector in self.detectors.items():
            if name != 'lof':
                try:
                    detector.fit(features_pca)
                    print(f"Fitted {name}")
                except Exception as e:
                    print(f"Failed to fit {name}: {e}")
                    # Remove failed detector
                    del self.detectors[name]
        
        # Store drift baseline (mean and std of recent normal samples)
        self.drift_baseline = {
            'mean': np.mean(features_pca, axis=0),
            'std': np.std(features_pca, axis=0)
        }
        
        return features_pca
    
    def detect_anomalies(self, features_pca):
        """
        Detect anomalies using a combination of models and identify the top 6 outliers.
        """
        print("Detecting anomalies...")
        
        anomaly_scores = {}
        n_samples = features_pca.shape[0]
        
        # 1. Isolation Forest
        if 'isolation_forest' in self.detectors:
            try:
                scores = self.detectors['isolation_forest'].decision_function(features_pca)
                # Scores are inverted (lower is more anomalous), so we flip them
                anomaly_scores['isolation_forest'] = -scores
                print(f"Calculated scores for isolation_forest")
            except Exception as e:
                print(f"Error with isolation_forest: {e}")

        # 2. Local Outlier Factor
        if 'lof' in self.detectors:
            try:
                self.detectors['lof'].fit(features_pca)
                # Higher score is more anomalous
                anomaly_scores['lof'] = -self.detectors['lof'].negative_outlier_factor_
                print(f"Calculated scores for lof")
            except Exception as e:
                print(f"Error with lof: {e}")

        # 3. KNN
        if 'knn_distance' in self.detectors:
            # k-NN distance approach for unsupervised anomaly detection.
            # The principle is that normal points lie in dense regions (small distance to neighbors),
            # while anomalies are isolated (large distance to neighbors).
            distances, indices = self.detectors['knn_distance'].kneighbors(features_pca)
            # Calculate the mean distance to the k-1 nearest neighbors.
            # We use distances[:, 1:] to exclude the first neighbor, which is the point itself (distance=0).
            anomaly_scores['knn_distance'] = np.mean(distances[:, 1:], axis=1)

        # 4. Mahalanobis Distance (Global)
        try:
            mean_pca = np.mean(features_pca, axis=0)
            cov_pca = np.cov(features_pca.T)
            try:
                inv_cov = np.linalg.inv(cov_pca)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_pca)
            
            mahal_distances = []
            for i in range(n_samples):
                diff = features_pca[i] - mean_pca
                mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                mahal_distances.append(mahal_dist)
            
            anomaly_scores['mahalanobis'] = np.array(mahal_distances)
            print(f"Calculated scores for mahalanobis")
        except Exception as e:
            print(f"Error with mahalanobis: {e}")
        
        # # 5. gaussion_mixture
        # if 'gaussian_mixture' in self.detectors: 
        #     detector = self.detectors['gaussian_mixture']
        #     log_likelihoods = detector.score_samples(features_pca)
        #     anomaly_scores['gaussian_mixture'] = -log_likelihoods  
          
        # Combine scores
        if not anomaly_scores:
            print("No anomaly scores were calculated.")
            return np.ones(n_samples), np.zeros(n_samples), {}, {}

        # Normalize scores to a common scale (0-1)
        normalized_scores = []
        for name, scores in anomaly_scores.items():
            if len(scores) > 0:
                score_range = np.max(scores) - np.min(scores)
                if score_range > 0:
                    norm_scores = (scores - np.min(scores)) / score_range
                else:
                    norm_scores = np.zeros_like(scores)
                normalized_scores.append(norm_scores)
        
        # Calculate final weighted score (simple average for fairness)
        if normalized_scores:
            final_scores = np.mean(normalized_scores, axis=0)
        else:
            final_scores = np.zeros(n_samples)

      

        # Print top 6 scores for each model
        print("\n--- Top 6 Anomaly Scores per Model ---")
        anomaly_indices = set()
        for model_name, scores in anomaly_scores.items():
            top6_indices = np.argsort(scores)[-6:]
            top6_scores = scores[top6_indices]
            anomaly_indices.update(top6_indices)
            # Reverse for descending order of scores in printout
            print(f"Model: {model_name}")
            print(f"  - Top 6 indices: {top6_indices[::-1]}")
            print(f"  - Top 6 scores:  {np.round(top6_scores[::-1], 4)}")
        print("------------------------------------")
        normal_indices = list(set(range(52)) - anomaly_indices)
        anomaly_indices = list(anomaly_indices)
        # Create predictions array (-1 for anomaly, 1 for normal)
        predictions = np.ones(n_samples)
        predictions[anomaly_indices] = -1
        
        print(f"Top 6 anomalies identified at indices: {anomaly_indices}")
        print(f"Top 6 weighted scores: {final_scores[anomaly_indices]}")

        return predictions, final_scores, normal_indices
    
    def visualize_results(self, features_pca, predictions, scores, sample_indices=None):
        """
        Visualize anomaly detection results
        """
        print("Creating visualizations...")
        
        if sample_indices is None:
            sample_indices = np.arange(len(predictions))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA scatter plot
        ax1 = axes[0, 0]
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        ax1.scatter(features_pca[normal_mask, 0], features_pca[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Normal', s=50)
        ax1.scatter(features_pca[anomaly_mask, 0], features_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, label='Anomaly', s=100, marker='X')
        
        # Add index labels to each point for better identification
        for i, index in enumerate(sample_indices):
            if i == 0:
                ax1.text(features_pca[i, 0] + 10, features_pca[i, 1] + 20, f' {index + 1}', 
                     fontsize=8, color='black', ha='left', va='center')
            elif normal_mask[i]:
                continue
            else:
                ax1.text(features_pca[i, 0] + 10, features_pca[i, 1] + 10, f' {index + 1}', 
                     fontsize=8, color='black', ha='left', va='center')
            
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.set_title('PCA Visualization of Anomalies')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Anomaly scores over time
        ax2 = axes[0, 1]
        ax2.plot(sample_indices, scores, 'b-', alpha=0.7, linewidth=2)
        ax2.scatter(sample_indices[anomaly_mask], scores[anomaly_mask], 
                   c='red', s=100, marker='X', label='Detected Anomalies')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Anomaly Scores Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
         # 1. PCA scatter plot
        # ax2 = axes[0, 1]
        # normal_mask = predictions == 1
        # anomaly_mask = predictions == -1
        
        # ax2.scatter(features_pca[normal_mask, 0], features_pca[normal_mask, 2], 
        #            c='blue', alpha=0.6, label='Normal', s=50)
        # ax2.scatter(features_pca[anomaly_mask, 0], features_pca[anomaly_mask, 2], 
        #            c='red', alpha=0.8, label='Anomaly', s=100, marker='X')
        
        # # Add index labels to each point for better identification
        # for i, index in enumerate(sample_indices):
        #     ax2.text(features_pca[i, 0], features_pca[i, 2], f' {index + 1}', 
        #              fontsize=8, color='black', ha='left', va='center')
            
        # ax2.set_xlabel('First Principal Component')
        # ax2.set_ylabel('Second Principal Component')
        # ax2.set_title('PCA Visualization of Anomalies')
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)

        
        # 3. Detection timeline
        ax3 = axes[1, 0]
        detection_timeline = predictions.copy()
        detection_timeline[detection_timeline == 1] = 0
        detection_timeline[detection_timeline == -1] = 1
        
        ax3.plot(sample_indices, detection_timeline, 'r-', linewidth=3, marker='o', markersize=8)
        ax3.fill_between(sample_indices, detection_timeline, alpha=0.3, color='red')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Anomaly Detection (1=Anomaly)')
        ax3.set_title('Anomaly Detection Timeline')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (PCA components)
        ax4 = axes[1, 1]
        explained_var_ratio = self.pca.explained_variance_ratio_[:10]  # Top 10 components
        component_indices = np.arange(1, len(explained_var_ratio) + 1)
        
        bars = ax4.bar(component_indices, explained_var_ratio, alpha=0.7, color='steelblue')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.set_title('Top 10 Principal Components')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('forward_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


# The known ground truth has 6 anomalies out of 52 samples. 6/52 \u2248 0.115
DEFAULT_CONTAMINATION_RATE = 6 / 52

def main_anomaly_detection(file_path, contamination_rate=DEFAULT_CONTAMINATION_RATE, pca_components=0.95):
    """
    Main function to run the complete anomaly detection pipeline
    """
    print("=== Semiconductor Wafer Anomaly Detection System ===")
    print(f"Expected contamination rate: {contamination_rate:.3f}")
    
    # Initialize detector
    detector = WaferAnomalyDetector(contamination=contamination_rate, pca_components=pca_components)
    
    # Load data (you'll need to implement the actual data loading)
    # data = load_and_process_wafer_data(file_path)
    
    print("\nTo use this system:")
    print("1. Ensure your data.csv is in the correct format")
    print("2. Each sample should be a d\u00d7p array (d=time points, p=129 sensors)")
    print("3. Modify the load_and_process_wafer_data function for your specific format")
    print("4. Run the detection pipeline")

    print("\nUsing 'flatten' feature extraction method (your suggestion).")
    # This method now returns pre-scaled features, so no extra scaling is needed.
    features_flattened = feature_engineering_01.run_flatten()

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_flattened)
    
    # Fit models on the (now scaled) features
    features_pca = detector.fit_drift_robust_model(features_scaled)
    
    # Detect anomalies
    predictions, scores, normal_indices = detector.detect_anomalies(features_pca)
    
    # Find anomaly indices
    anomaly_indices = np.where(predictions == -1)[0]
    print(f"Detected normal sample indices: {[i for i in normal_indices]}\n")
    
    # Visualize results
    detector.visualize_results(features_pca, predictions, scores)
    

# Example usage
if __name__ == "__main__":
    # You can now choose the feature extraction method:
    # 'statistical': The original method, loses temporal data but low-dim.
    # 'flatten': Your suggested method, preserves temporal data but high-dim.

    # You can also control PCA components:
    # - A float between 0.0 and 1.0 to keep a certain percentage of variance (e.g., 0.95)
    # - An integer to select a fixed number of top components (e.g., 10)
    detector, _, _, _ = main_anomaly_detection(
        "data/data.xlsx", 
        pca_components=10 # Example: use top 10 principal components
    )
    print("\nAnomalyDetection system initialized and ready to use!")
