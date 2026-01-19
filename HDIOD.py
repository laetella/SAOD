import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation
import time
import warnings
warnings.filterwarnings('ignore')

class HDIOD:
    """
    High-Density Iteration Outlier Detection (HDIOD) algorithm
    Based on the paper: "Outlier detection method based on high-density iteration"
    """
    
    def __init__(self, k=10, a=2.5, b=1.4826, max_iter=100):
        """
        Initialize HDIOD algorithm
        
        Parameters:
        -----------
        k : int, default=10
            Number of nearest neighbors
        a : float, default=2.5
            Parameter for threshold calculation
        b : float, default=1.4826
            Parameter for MAD calculation
        max_iter : int, default=100
            Maximum number of iterations to prevent infinite loops
        """
        self.k = k
        self.a = a
        self.b = b
        self.max_iter = max_iter
        self.local_densities = None
        self.knn_indices = None
        self.knn_distances = None
        self.cof_scores = None
        self.threshold = None
        
    def _compute_knn_manual(self, X):
        """
        Manual k-NN computation to avoid scikit-learn compatibility issues
        """
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = cdist(X, X)
        
        # Set diagonal to infinity to exclude self
        np.fill_diagonal(distances, np.inf)
        
        # Find k nearest neighbors
        knn_indices = np.zeros((n_samples, self.k), dtype=int)
        knn_distances = np.zeros((n_samples, self.k))
        
        for i in range(n_samples):
            # Get indices of k smallest distances
            idx = np.argpartition(distances[i], self.k)[:self.k]
            # Sort by distance
            sorted_idx = idx[np.argsort(distances[i][idx])]
            knn_indices[i] = sorted_idx
            knn_distances[i] = distances[i][sorted_idx]
            
        return knn_indices, knn_distances
    
    def _gaussian_kernel(self, distances, d):
        """
        Compute Gaussian kernel function
        """
        # Add small epsilon to avoid numerical issues
        distances = np.maximum(distances, 1e-10)
        return 1 / ((2 * np.pi) ** (d / 2)) * np.exp(-distances**2 / 2)
    
    def _compute_local_kernel_density(self, X):
        """
        Compute local kernel density for each sample using Gaussian kernel and k-NN
        """
        n_samples, n_features = X.shape
        
        # Compute k-NN manually
        self.knn_indices, self.knn_distances = self._compute_knn_manual(X)
        
        local_densities = np.zeros(n_samples)
        
        for i in range(n_samples):
            if len(self.knn_distances[i]) > 0:
                # Get distances to k-nearest neighbors
                distances = self.knn_distances[i]
                
                # Compute Gaussian kernel values
                kernel_values = self._gaussian_kernel(distances, n_features)
                
                # Compute local kernel density (Eq. 3 in paper)
                local_densities[i] = np.mean(kernel_values)
            else:
                # Single sample case
                local_densities[i] = 1.0
                
        return local_densities
    
    def _high_density_iteration(self, X):
        """
        Perform high-density iteration for all samples
        """
        n_samples = X.shape[0]
        iteration_matrices = []
        extended_knn_sets = []
        
        for i in range(n_samples):
            iteration_path = [i]  # Start with current sample
            current_sample = i
            iteration_count = 0
            
            while iteration_count < self.max_iter:
                iteration_count += 1
                
                # Get k-nearest neighbors of current sample
                if len(self.knn_indices[current_sample]) == 0:
                    break  # No neighbors available
                    
                current_knn = self.knn_indices[current_sample]
                
                # Get local densities of neighbors
                neighbor_densities = self.local_densities[current_knn]
                
                # Check if current sample's density is less than max neighbor density
                current_density = self.local_densities[current_sample]
                max_neighbor_density = np.max(neighbor_densities)
                
                if current_density >= max_neighbor_density:
                    # Termination condition met
                    break
                else:
                    # Find neighbor with highest density
                    max_density_idx = np.argmax(neighbor_densities)
                    next_sample = current_knn[max_density_idx]
                    
                    # Avoid infinite loops by checking if we're revisiting samples
                    if next_sample in iteration_path:
                        break
                    
                    iteration_path.append(next_sample)
                    current_sample = next_sample
            
            iteration_matrices.append(iteration_path)
            
            # Build extended k-nearest neighbors set (Definition 2)
            extended_knn = set()
            for sample_idx in iteration_path:
                if len(self.knn_indices[sample_idx]) > 0:
                    extended_knn.update(self.knn_indices[sample_idx])
            extended_knn_sets.append(list(extended_knn))
            
        return iteration_matrices, extended_knn_sets
    
    def _compute_centripetal_outlier_factor(self, iteration_matrices, extended_knn_sets):
        """
        Compute centripetal outlier factor for each sample
        """
        n_samples = len(iteration_matrices)
        cof_scores = np.ones(n_samples)  # Default to 1
        
        for i in range(n_samples):
            iteration_path = iteration_matrices[i]
            extended_knn = extended_knn_sets[i]
            
            if len(extended_knn) == 0:
                # No extended neighbors, use current density
                cof_scores[i] = 1.0
                continue
                
            # Maximum local kernel density in extended k-nearest neighbors
            max_density_in_extended = np.max(self.local_densities[extended_knn])
            
            # Current sample's local density
            current_density = self.local_densities[i]
            
            # Compute centripetal outlier factor (Eq. 6 in paper)
            if current_density > 1e-10:  # Avoid division by zero
                cof_scores[i] = max_density_in_extended / current_density
            else:
                cof_scores[i] = 1e10  # Very large value for zero density
                
        return cof_scores
    
    def _compute_threshold(self, scores):
        """
        Compute threshold using median and MAD (Median Absolute Deviation)
        """
        if len(scores) == 0:
            return 0.0
            
        median_score = np.median(scores)
        
        # Handle MAD calculation safely
        try:
            mad = self.b * median_abs_deviation(scores, scale='normal')
        except:
            # Fallback: use standard deviation if MAD fails
            mad = self.b * np.std(scores)
            
        threshold = median_score + self.a * mad
        return threshold
    
    def fit(self, X):
        """
        Fit the HDIOD model
        """
        X = np.asarray(X, dtype=np.float64)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Input data is empty")
            
        # Adjust k if necessary
        self.k = min(self.k, n_samples - 1)
        if self.k < 1:
            self.k = 1
            
        # Step 1: Compute local kernel density
        self.local_densities = self._compute_local_kernel_density(X)
        
        # Step 2: Perform high-density iteration
        self.iteration_matrices, self.extended_knn_sets = self._high_density_iteration(X)
        
        # Step 3: Compute centripetal outlier factor
        self.cof_scores = self._compute_centripetal_outlier_factor(
            self.iteration_matrices, self.extended_knn_sets
        )
        
        # Step 4: Compute threshold
        self.threshold = self._compute_threshold(self.cof_scores)
        
        return self
    
    def predict(self, X=None):
        """
        Predict outlier labels
        """
        if X is not None:
            # For new data, we need to recompute everything
            self.fit(X)
            
        if self.cof_scores is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        labels = (self.cof_scores > self.threshold).astype(int)
        return labels
    
    def fit_predict(self, X):
        """
        Fit the model and predict outlier labels
        """
        return self.fit(X).predict()
    
    def decision_function(self, X=None):
        """
        Compute outlier scores
        """
        if X is not None:
            # For new data, we need to recompute everything
            self.fit(X)
            
        if self.cof_scores is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.cof_scores

def generate_test_data():
    """Generate test data for demonstration"""
    np.random.seed(42)
    
    # Generate normal data (3 clusters)
    n_normal = 300
    centers = np.array([[1, 1], [5, 5], [8, 1]])
    cluster_std = 0.5
    
    X_normal = []
    for center in centers:
        cluster_data = np.random.randn(n_normal // 3, 2) * cluster_std + center
        X_normal.append(cluster_data)
    X_normal = np.vstack(X_normal)
    
    # Generate outliers
    n_outliers = 30
    X_outliers = np.random.uniform(low=-2, high=11, size=(n_outliers, 2))
    
    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_outliers))])
    
    return X, y

def demonstrate_hdiod():
    """Demonstrate the HDIOD algorithm"""
    
    print("Generating test data...")
    X, y_true = generate_test_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of true outliers: {np.sum(y_true)}")
    
    # Apply HDIOD
    print("\nRunning HDIOD algorithm...")
    hdiod = HDIOD(k=15)
    
    start_time = time.time()
    hdiod.fit(X)
    y_pred = hdiod.predict()
    end_time = time.time()
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, f1_score
    
    auc_score = roc_auc_score(y_true, hdiod.cof_scores)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nHDIOD Results:")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Number of detected outliers: {np.sum(y_pred)}")
    print(f"Threshold: {hdiod.threshold:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data with true labels
    plt.subplot(1, 3, 1)
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1], c='blue', alpha=0.6, label='Normal', s=30)
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], c='red', alpha=0.6, label='True Outliers', s=50)
    plt.title('True Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # HDIOD results
    plt.subplot(1, 3, 2)
    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', alpha=0.6, label='Predicted Normal', s=30)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', alpha=0.6, label='Predicted Outliers', s=50)
    plt.title('HDIOD Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Outlier scores
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=hdiod.cof_scores, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Centripetal Outlier Factor')
    plt.title('Outlier Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return hdiod, X, y_true, y_pred

def analyze_iteration_examples(hdiod, X, n_examples=3):
    """Analyze iteration process for example points"""
    
    # Find examples of different types: normal, boundary, outlier
    scores = hdiod.cof_scores
    normal_idx = np.argsort(scores)[:n_examples]  # Lowest scores (normal)
    outlier_idx = np.argsort(scores)[-n_examples:]  # Highest scores (outliers)
    boundary_idx = np.argsort(scores)[len(scores)//2:len(scores)//2 + n_examples]  # Middle scores
    
    all_examples = list(normal_idx) + list(boundary_idx) + list(outlier_idx)
    labels = ['Normal'] * n_examples + ['Boundary'] * n_examples + ['Outlier'] * n_examples
    
    plt.figure(figsize=(15, 4 * n_examples))
    
    for i, (idx, label) in enumerate(zip(all_examples, labels)):
        plt.subplot(3, n_examples, i + 1)
        
        # Plot all points
        plt.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.3, s=20)
        
        # Highlight the iteration path
        iteration_path = hdiod.iteration_matrices[idx]
        path_coords = X[iteration_path]
        
        # Plot iteration path
        plt.plot(path_coords[:, 0], path_coords[:, 1], 'ro-', linewidth=2, markersize=6)
        plt.scatter(path_coords[0, 0], path_coords[0, 1], c='green', s=100, label='Start')
        plt.scatter(path_coords[-1, 0], path_coords[-1, 1], c='blue', s=100, label='End')
        
        # Plot k-nearest neighbors
        if len(hdiod.knn_indices[idx]) > 0:
            knn_indices = hdiod.knn_indices[idx]
            plt.scatter(X[knn_indices, 0], X[knn_indices, 1], c='orange', s=50, alpha=0.7, label='k-NN')
        
        plt.title(f'{label} (Sample {idx})\nCOF = {hdiod.cof_scores[idx]:.3f}, Iterations = {len(iteration_path)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def parameter_sensitivity_analysis(X, y_true):
    """Analyze sensitivity to parameter k"""
    
    k_values = [5, 10, 15, 20, 25, 30]
    auc_scores = []
    f1_scores = []
    
    print("Parameter sensitivity analysis...")
    
    for k in k_values:
        hdiod = HDIOD(k=k)
        hdiod.fit(X)
        y_pred = hdiod.predict()
        
        auc = roc_auc_score(y_true, hdiod.cof_scores)
        f1 = f1_score(y_true, y_pred)
        
        auc_scores.append(auc)
        f1_scores.append(f1)
        
        print(f"k={k}: AUC={auc:.4f}, F1={f1:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, auc_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('AUC Score')
    plt.title('AUC vs k')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, f1_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs k')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return k_values, auc_scores, f1_scores

if __name__ == "__main__":
    print("HDIOD Algorithm - Complete Implementation")
    print("=" * 50)
    
    try:
        # Main demonstration
        hdiod, X, y_true, y_pred = demonstrate_hdiod()
        
        # Analyze iteration process
        print("\nAnalyzing iteration process...")
        analyze_iteration_examples(hdiod, X)
        
        # Parameter sensitivity
        print("\nPerforming parameter sensitivity analysis...")
        parameter_sensitivity_analysis(X, y_true)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()