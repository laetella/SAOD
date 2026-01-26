import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
import time

class ADRN:
    def __init__(self, alpha=0.26, m=None, max_iter=100, tol=1e-6, 
                 w_max_iter=50, s_max_iter=50, step_size_w=0.01, step_size_s=0.01):
        """
        ADRN: Anomaly Detection with Representative Neighbors
        
        Parameters:
        -----------
        alpha : float, regularization parameter for l2,1 norm
        m : int, dimension of projected space (if None, auto set)
        max_iter : int, maximum iterations for main loop
        tol : float, convergence tolerance
        w_max_iter : int, maximum iterations for W update
        s_max_iter : int, maximum iterations for S update
        step_size_w : float, step size for W gradient descent
        step_size_s : float, step size for S gradient descent
        """
        self.alpha = alpha
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.w_max_iter = w_max_iter
        self.s_max_iter = s_max_iter
        self.step_size_w = step_size_w
        self.step_size_s = step_size_s
        
        self.S = None
        self.W = None
        self.objective_values = []
        
    def _initialize_parameters(self, X):
        """Initialize S and W according to paper constraints"""
        n, p = X.shape
        
        if self.m is None:
            self.m = min(p, 100)  # Default projection dimension
            
        # Initialize W with random normal values
        self.W = np.random.randn(p, self.m) * 0.01
        
        # Initialize S with uniform distribution satisfying constraints
        self.S = np.random.rand(n, n)
        np.fill_diagonal(self.S, 0)  # s_ii = 0
        # Normalize rows to sum to 1
        row_sums = self.S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.S = self.S / row_sums
        
    def _objective_function(self, X, S, W, D=None):
        """Calculate objective function value from Eq.(4)"""
        if D is None:
            D = self._calculate_D(W)
            
        reconstruction_error = S @ X @ W - X @ W
        term1 = np.linalg.norm(reconstruction_error, 'fro') ** 2
        term2 = self.alpha * np.trace(W.T @ D @ W)
        
        return term1 + term2
    
    def _calculate_D(self, W):
        """Calculate diagonal matrix D from Eq.(8)"""
        p, m = W.shape
        D = np.zeros((p, p))  # D should be p x p for W: p x m
        
        for i in range(p):
            w_norm = np.linalg.norm(W[i, :])
            if w_norm > 1e-10:
                D[i, i] = 1.0 / (2 * w_norm)
            # else remains 0
            
        return D
    
    def _update_W_fixed_S(self, X, S):
        """
        Update W while fixing S (Algorithm 1 in paper)
        Using iteratively reweighted least squares approach
        """
        n, p = X.shape
        W_old = self.W.copy()
        
        for t in range(self.w_max_iter):
            # Calculate current D matrix (p x p)
            D = self._calculate_D(W_old)
            
            # Calculate gradient from Eq.(6)
            # Note: The paper's equation might have dimensionality issues
            # Let's use a more stable formulation
            XW = X @ W_old
            SXW = S @ XW
            
            # Gradient computation
            grad_part1 = X.T @ (SXW - XW)  # p x m
            grad_part2 = X.T @ (S.T @ (SXW - XW))  # p x m
            grad = 2 * (grad_part1 + grad_part2) + self.alpha * D @ W_old
            
            # Alternative simpler gradient (more stable)
            # M = S - np.eye(n)
            # grad_simple = 2 * X.T @ M.T @ M @ X @ W_old + self.alpha * D @ W_old
            
            # Update W using gradient descent (Eq.7)
            W_new = W_old - self.step_size_w * grad
            
            # Check convergence for W
            if np.linalg.norm(W_new - W_old, 'fro') < self.tol:
                break
                
            W_old = W_new
            
        return W_new
    
    def _update_S_fixed_W(self, X, W):
        """
        Update S while fixing W using reduced gradient method
        Based on Eq.(9-11) in paper
        """
        n = X.shape[0]
        XW = X @ W
        S_new = self.S.copy()
        
        for i in range(n):
            s_i = S_new[i, :].copy()
            
            # Find index of largest nonzero entry (m in paper)
            non_zero_mask = (s_i > 1e-10) & (np.arange(n) != i)
            if np.sum(non_zero_mask) == 0:
                # If all zeros, initialize uniformly (except diagonal)
                s_i = np.ones(n) / (n - 1)
                s_i[i] = 0
                S_new[i, :] = s_i
                continue
                
            m_idx = np.argmax(s_i)
            
            # Calculate the residual
            residual = S_new @ XW - XW
            
            # Calculate partial derivatives ∂f/∂s_ik
            partial_deriv = np.zeros(n)
            for k in range(n):
                if k != i:
                    # ∂f/∂s_ik = 2 * (SXW - XW)[i] · XW[k]
                    partial_deriv[k] = 2 * np.dot(residual[i, :], XW[k, :])
            
            # Calculate reduced gradient components (Eq.10)
            grad = np.zeros(n)
            for k in range(n):
                if k != i and k != m_idx:
                    if s_i[k] > 1e-10 or (s_i[k] <= 1e-10 and (partial_deriv[k] - partial_deriv[m_idx]) <= 0):
                        grad[k] = partial_deriv[k] - partial_deriv[m_idx]
                    else:
                        grad[k] = 0
            
            # Calculate gradient for m_idx (Eq.11)
            grad[m_idx] = -np.sum([grad[k] for k in range(n) if k != m_idx])
            
            # Update s_i along gradient descent direction
            s_i_new = s_i - self.step_size_s * grad
            
            # Project to feasible set: s_ii = 0, s_ij >= 0, sum(s_i) = 1
            s_i_new[i] = 0
            s_i_new = np.maximum(s_i_new, 0)
            
            # Renormalize to satisfy sum constraint
            sum_s = np.sum(s_i_new)
            if sum_s < 1e-10:
                # If all entries become zero, reinitialize uniformly
                s_i_new = np.ones(n) / (n - 1)
                s_i_new[i] = 0
            else:
                s_i_new = s_i_new / sum_s
            
            S_new[i, :] = s_i_new
            
        return S_new
    
    def fit(self, X, verbose=False):
        """
        Main training procedure for ADRN
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        verbose : bool, whether to print progress
        """
        n, p = X.shape
        self._initialize_parameters(X)
        
        prev_obj = float('inf')
        
        for iter_idx in range(self.max_iter):
            start_time = time.time()
            
            # Update W with fixed S
            self.W = self._update_W_fixed_S(X, self.S)
            
            # Update S with fixed W
            self.S = self._update_S_fixed_W(X, self.W)
            
            # Calculate current objective value
            current_obj = self._objective_function(X, self.S, self.W)
            self.objective_values.append(current_obj)
            
            # Check convergence (Proposition 1 in paper)
            if iter_idx > 0:
                obj_diff = prev_obj - current_obj
                if verbose:
                    print(f"Iteration {iter_idx+1}: Objective = {current_obj:.6f}, "
                          f"Difference = {obj_diff:.6f}, Time = {time.time()-start_time:.2f}s")
                
                if obj_diff < self.tol and obj_diff >= 0:
                    if verbose:
                        print(f"Converged at iteration {iter_idx+1}")
                    break
            else:
                if verbose:
                    print(f"Iteration {iter_idx+1}: Objective = {current_obj:.6f}, "
                          f"Time = {time.time()-start_time:.2f}s")
                
            prev_obj = current_obj
    
    def calculate_similarity_matrix(self):
        """
        Calculate similarity matrix using cosine metric (Eq.12-13)
        """
        n = self.S.shape[0]
        S_sim = np.zeros((n, n))
        
        for i in range(n):
            s_i = self.S[i, :]
            norm_i = np.linalg.norm(s_i)
            
            for j in range(n):
                if i == j:
                    S_sim[i, j] = 1.0
                else:
                    s_j = self.S[j, :]
                    norm_j = np.linalg.norm(s_j)
                    
                    if norm_i < 1e-10 or norm_j < 1e-10:
                        S_sim[i, j] = 0.0
                    else:
                        # Cosine similarity (Eq.12)
                        cos_sim = np.dot(s_i, s_j) / (norm_i * norm_j)
                        
                        # Rectify negative values (Eq.13)
                        S_sim[i, j] = max(0, cos_sim)
        
        return S_sim
    
    def detect_anomalies(self, X, k=None, return_scores=False):
        """
        Detect anomalies using graph clustering approach (Algorithm 2)
        
        Parameters:
        -----------
        X : array-like, input data
        k : int, number of anomalies to return (if None, auto determine)
        return_scores : bool, whether to return anomaly scores
        
        Returns:
        --------
        anomaly_indices : array, indices of detected anomalies
        anomaly_scores : array, anomaly scores (if return_scores=True)
        """
        if self.S is None:
            self.fit(X)
        
        # Calculate similarity matrix
        S_sim = self.calculate_similarity_matrix()
        
        # Calculate diagonal matrix H and Laplacian L
        H = np.diag(np.sum(S_sim, axis=1))
        L = H - S_sim
        
        # Solve generalized eigenvalue problem: Lv = λHv (Eq.14)
        n = X.shape[0]
        try:
            # Use dense eigensolver
            eigvals, eigvecs = eigh(L, H)
            
            # Find the second smallest eigenvalue and corresponding eigenvector
            # Skip the smallest eigenvalue which is usually 0
            sorted_indices = np.argsort(eigvals)
            second_smallest_idx = sorted_indices[1] if n > 1 else sorted_indices[0]
            v = eigvecs[:, second_smallest_idx].real
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback: use simple spectral clustering
            print("Warning: Using fallback eigenvalue solver")
            eigvals, eigvecs = eigh(L)
            sorted_indices = np.argsort(eigvals)
            second_smallest_idx = sorted_indices[1] if n > 1 else sorted_indices[0]
            v = eigvecs[:, second_smallest_idx].real
        
        # Use coefficients as anomaly scores (smaller = more anomalous)
        anomaly_scores = np.abs(v)  # Use absolute values for stability
        
        if k is None:
            # Auto-determine k based on score distribution
            k = max(1, n // 20)  # Default to 5%
        
        # Get top k anomalies (smallest coefficients)
        anomaly_indices = np.argsort(anomaly_scores)[:k]
        
        if return_scores:
            return anomaly_indices, anomaly_scores
        else:
            return anomaly_indices
    
    def get_objective_history(self):
        """Return history of objective function values"""
        return self.objective_values


# Test the implementation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Generate synthetic dataset with anomalies
    np.random.seed(42)
    
    print("Generating synthetic dataset...")
    # Normal data
    X_normal, _ = make_blobs(n_samples=190, centers=2, n_features=50, 
                            cluster_std=1.0, random_state=42)
    
    # Anomalous data (different distribution)
    X_anomaly = np.random.multivariate_normal(
        mean=[5] * 50, 
        cov=np.eye(50) * 3.0, 
        size=10
    )
    
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0] * 190 + [1] * 10)  # 1 indicates anomaly
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset shape: {X_scaled.shape}")
    print(f"Number of true anomalies: {np.sum(y_true)}")
    
    # Initialize and fit ADRN
    print("\nFitting ADRN model...")
    adrn = ADRN(alpha=0.1, m=20, max_iter=30, tol=1e-6, 
                step_size_w=0.001, step_size_s=0.01)
    
    try:
        adrn.fit(X_scaled, verbose=True)
        
        # Detect anomalies
        print("\nDetecting anomalies...")
        anomaly_indices, anomaly_scores = adrn.detect_anomalies(X_scaled, k=10, return_scores=True)
        
        # Create predicted labels
        y_pred = np.zeros(len(X))
        y_pred[anomaly_indices] = 1
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nDetection Results:")
        print(f"True anomalies: {np.where(y_true == 1)[0]}")
        print(f"Detected anomalies: {anomaly_indices}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        # Plot objective function convergence
        plt.subplot(1, 3, 1)
        plt.plot(adrn.get_objective_history())
        plt.title('Objective Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid(True)
        
        # Plot anomaly scores
        plt.subplot(1, 3, 2)
        colors = ['blue' if label == 0 else 'red' for label in y_true]
        plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=colors, alpha=0.6)
        plt.title('Anomaly Scores\n(Red=True Anomalies)')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        
        # Plot detected vs true anomalies
        plt.subplot(1, 3, 3)
        true_positives = np.intersect1d(np.where(y_true == 1)[0], anomaly_indices)
        false_positives = np.setdiff1d(anomaly_indices, np.where(y_true == 1)[0])
        false_negatives = np.setdiff1d(np.where(y_true == 1)[0], anomaly_indices)
        
        all_indices = range(len(X))
        normal_indices = np.setdiff1d(all_indices, np.concatenate([true_positives, false_positives, false_negatives]))
        
        plt.scatter(normal_indices, [0] * len(normal_indices), c='blue', label='Normal', alpha=0.6)
        plt.scatter(true_positives, [0] * len(true_positives), c='green', label='True Positive', alpha=0.8)
        plt.scatter(false_positives, [0] * len(false_positives), c='orange', label='False Positive', alpha=0.8)
        plt.scatter(false_negatives, [0] * len(false_negatives), c='red', label='False Negative', alpha=0.8)
        plt.title('Detection Results')
        plt.xlabel('Sample Index')
        plt.yticks([])
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print algorithm statistics
        print(f"\nAlgorithm Statistics:")
        print(f"Final objective value: {adrn.get_objective_history()[-1]:.6f}")
        print(f"Number of iterations: {len(adrn.get_objective_history())}")
        print(f"S matrix sparsity: {np.mean(adrn.S < 1e-6):.3f}")
        print(f"W matrix shape: {adrn.W.shape}")
        print(f"S matrix shape: {adrn.S.shape}")
        
    except Exception as e:
        print(f"Error during fitting: {e}")
        import traceback
        traceback.print_exc()