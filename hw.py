import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

def exact_solution(x):
    """
    Exact solution u(x) = sin(pi*x)
    Corresponding to the right-hand side f(x) = pi^2 * sin(pi*x)
    """
    return np.sin(np.pi * x * 3)

def source_term(x):
    """
    The right-hand side of the equation f(x) = pi^2 * sin(pi*x)
    """
    return ((3*np.pi)**2) * np.sin(3*np.pi * x)

def solve_fem_1d(n):
    """
    Solve the 1D boundary value problem using finite element method
    -u''(x) = f(x), x in [0,1]
    u(0) = u(1) = 0
    
    Parameters:
        n: Number of subintervals
    
    Returns:
        x_nodes: Node coordinates
        u_h: Finite element solution
        u_exact: Exact solution at nodes
    """
    # Step size
    h = 1.0 / n
    
    # Create nodes
    x_nodes = np.linspace(0, 1, n+1)
    
    # Build stiffness matrix and load vector
    A = lil_matrix((n+1, n+1))
    b = np.zeros(n+1)
    
    # Assemble stiffness matrix and load vector
    for i in range(n):
        # Interval [x_i, x_{i+1}]
        x_left = x_nodes[i]
        x_right = x_nodes[i+1]
        
        # Element stiffness matrix
        A[i, i] += 1.0 / h
        A[i, i+1] += -1.0 / h
        A[i+1, i] += -1.0 / h
        A[i+1, i+1] += 1.0 / h
        
        # Element load vector (using trapezoidal rule for integration)
        f_left = source_term(x_left)
        f_right = source_term(x_right)
        b[i] += h * (f_left / 2)
        b[i+1] += h * (f_right / 2)
    
    # Apply boundary conditions
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = 0
    
    A[n, :] = 0
    A[n, n] = 1
    b[n] = 0
    
    # Convert to CSR format for efficient solving
    A = csr_matrix(A)
    
    # Solve the linear system
    u_h = spsolve(A, b)
    
    # Calculate the exact solution
    u_exact = exact_solution(x_nodes)
    
    return x_nodes, u_h, u_exact

def compute_errors(n):
    """
    Compute the errors of the finite element solution
    
    Parameters:
        n: Number of subintervals
    
    Returns:
        l2_error: L2 error
        h1_error: H1 seminorm error
    """
    x_nodes, u_h, u_exact = solve_fem_1d(n)
    h = 1.0 / n
    
    # Calculate L2 error
    l2_error = np.sqrt(h * np.sum((u_h - u_exact)**2))
    
    # Calculate H1 seminorm error (L2 error of the derivative)
    h1_seminorm_error = 0
    
    # Derivative of the exact solution
    du_exact = lambda x: np.pi * np.cos(np.pi * x)
    
    for i in range(n):
        # The derivative of the finite element solution is constant in each element
        du_h = (u_h[i+1] - u_h[i]) / h
        
        # Evaluate the derivative of the exact solution at the midpoint
        x_mid = (x_nodes[i] + x_nodes[i+1]) / 2
        du_exact_mid = du_exact(x_mid)
        
        # Accumulate the error
        h1_seminorm_error += h * (du_h - du_exact_mid)**2
    
    h1_seminorm_error = np.sqrt(h1_seminorm_error)
    
    return l2_error, h1_seminorm_error

def convergence_study():
    """
    Perform a convergence study to verify the convergence rate of errors
    """
    # Different mesh sizes
    n_values = [8, 16, 32, 64, 128, 256]
    h_values = [1.0/n for n in n_values]
    
    # Store errors and computation times
    l2_errors = []
    h1_errors = []
    times = []
    
    for n in n_values:
        start_time = time.time()
        l2_error, h1_error = compute_errors(n)
        end_time = time.time()
        
        l2_errors.append(l2_error)
        h1_errors.append(h1_error)
        times.append(end_time - start_time)
        
        print(f"Grid size n = {n}:")
        print(f"  h = {1.0/n:.6f}")
        print(f"  L2 error = {l2_error:.6e}")
        print(f"  H1 seminorm error = {h1_error:.6e}")
        print(f"  Computation time = {times[-1]:.6f} seconds")
        
        # Calculate convergence rate if not the first grid
        if len(l2_errors) > 1:
            l2_rate = np.log(l2_errors[-2]/l2_errors[-1]) / np.log(2)
            h1_rate = np.log(h1_errors[-2]/h1_errors[-1]) / np.log(2)
            print(f"  L2 error convergence rate = {l2_rate:.4f}")
            print(f"  H1 seminorm error convergence rate = {h1_rate:.4f}")
        
        print("")
    
    # Plot error convergence
    plt.figure(figsize=(10, 8))
    
    plt.loglog(h_values, l2_errors, 'o-', label='L2 error')
    plt.loglog(h_values, h1_errors, 's-', label='H1 seminorm error')
    
    # Add reference lines
    ref_h2 = [h**2 * l2_errors[0]/(h_values[0]**2) for h in h_values]
    ref_h1 = [h * h1_errors[0]/h_values[0] for h in h_values]
    
    plt.loglog(h_values, ref_h2, '--', label='O(h²) reference')
    plt.loglog(h_values, ref_h1, '-.', label='O(h) reference')
    
    plt.xlabel('Mesh size (h)')
    plt.ylabel('Error')
    plt.title('Finite Element Method Error Convergence Study')
    plt.legend()
    plt.grid(True)
    
    # Plot comparison of solutions
    plt.figure(figsize=(10, 6))
    
    # Choose a mesh size for illustration
    n_plot = 16
    x_nodes, u_h, u_exact = solve_fem_1d(n_plot)
    
    # Generate denser points for the exact solution
    x_dense = np.linspace(0, 1, 1000)
    u_exact_dense = exact_solution(x_dense)
    
    plt.plot(x_dense, u_exact_dense, 'r-', label='Exact solution')
    plt.plot(x_nodes, u_h, 'bo-', label=f'FEM solution (n={n_plot})')
    
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of Exact and Finite Element Solutions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the convergence study
    convergence_study()