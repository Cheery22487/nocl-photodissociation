import numpy as np
import time
import sys
import random
import copy

################################### INPUT ###################################

parameter = [
    {"type": "HO", "N": 24,"w": 0.008126, "shift": 2.155},
    {"type": "Fourier", "xN": 96, "x0": 3.5, "x_end": 8.3, "p0": 0},
    {"type": "Legendre", "N": 60}
    ]

N_lanczos = 11

################################### INPUT ###################################

np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)

## This function creates the initial wavefunction.
## The coefficients of the multidimensional wavefunction are intially all saved in a 1-D vector. Can be changed at any time using np.reshape later on
def generate_psi(N):
    vec = np.zeros(shape=(N))
    for x in range(len(vec)):
        vec[x] = random.randint(2,10)   #using a distribution of random values as the initial "guess" of the wave function helps the convergence speed of the ground state later
    vec = vec / np.linalg.norm(vec)     #wavefunctions must be normalized
    return vec


## This functions generates the matrices of the 1-D harmonic osciallor problem used to describe the harmonic oscillator degrees of freedom
def generate_ho_dvr(parameter):
    N = parameter["N"]                                                  #N: Amount of basisfunctions used
    x_mat = generate_x_mat(N)                                           #Generation the matrix representation of the position operator in the harmonic oscillator space
    t_mat = generate_t_mat(N)                                           #Generation the matrix representation of the kinetic energy operator in the harmonic oscillator space

    x_diag, x_eigen, x_eigen_tr, x_eigenvalues = diagonalize(x_mat)    #The resulting diagonal matrix is still the matrix representation of the position operator but in position space
                                                                        #Additionally the matrix of the yielded eigenvectors from the diagonalization is the transformation matrix from harmonic osciallator space to position space
    
    t_mat_nb = x_eigen_tr @ t_mat @ x_eigen / 4                         #Transforming the kinetic energy operator matrix to position space using the transformation matrix

    return t_mat_nb, x_eigenvalues / np.sqrt(2)                         

## Generation of the matrix representation of the position operator of the harmonic oscillator
def generate_x_mat(N):
    a = np.zeros(shape=(N,N))
    for n in range(len(a)):
        for m in range(len(a[n])):
            if m == n+1:
                a[n,m] = np.sqrt(n+1)   # The specific values can be derived using a linear combination of the ladder operators
            elif m == n-1:              #(https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#Ladder_operator_method)
                a[n,m] = np.sqrt(n)
    return a

## Generation of the matrix representation of the momentum operator of the harmonic oscillator
def generate_t_mat(N):
    a = np.zeros(shape=(N,N))
    for n in range(len(a)):
        for m in range(len(a[n])):
            if m == n:
                a[n,m] = -(2*n+1)
            elif m == n-2:
                a[n,m] = np.sqrt(n-1) * np.sqrt(n)  # The specific values can be derived using a linear combination of the ladder operators 
            elif m == n+2:                          #(https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#Ladder_operator_method)
                a[n,m] = np.sqrt(n+2) * np.sqrt(n+1)
    return a



## This functions generates the matrices using equidistant grids of position and momentum for the fourier degrees of freedom
def generate_fourier_dvr(parameter):
    x_N = parameter["xN"]                                               #Amount if gridpoints
    x_0 = parameter["x0"]                                               #Start of the grid. Minimum distance in the respective coordinate considered
    x_end = parameter["x_end"]                                          #End of the grid

    x_gitter, delta_x = generate_x_gitter_fourier(x_0,x_N,x_0,x_end)
    
    x_points = np.zeros(shape=(x_N))
    for i in range(len(x_gitter)):
        x_points[i] = x_gitter[i,i]                                     #Equidistant gridpoints of position between x_0 and x_end. Position space!
        
    p_gitter = generate_p_gitter(x_N,delta_x)                           #Equidistant gridpoints of momentum. Momentum space!
    xp_trans = gen_xp_trans(x_N,x_gitter,p_gitter)                      #Generates the transformation matrix between position space and momentum space

    t_mat = (p_gitter @ p_gitter) / 2

    t_dvr = xp_trans.conj() @ t_mat @ xp_trans.T                        #Transforming the kinetic energy operator matrix to position space

    return t_dvr, x_points

##Generation of the equidistant position grid
def generate_x_gitter_fourier(x_0,x_N,start,end):
    delta_x = (end-start)/(x_N-1)                       #distance between gridpoints
    a = np.zeros(shape=(x_N,x_N))
    for n in range(len(a)):
        a[n,n] = x_0 + delta_x * n

    return a, delta_x

##Generation of the equidistant momentum grid for fourier
def generate_p_gitter(x_N,delta_x):
    delta_p = 2*np.pi / (x_N * delta_x)                 #Distances of gridpoints determined by Delta_p * Delta_x = 2*pi/x_N

    p_0 = 0 - (x_N/2 - 0.5)*delta_p
    #p_0 = 0
    a = np.zeros(shape=(x_N,x_N))
    for n in range(len(a)):
        a[n,n] = p_0 + delta_p * n

    return a

##Generation of the transformation matrix between momentum space and position space (Source of the formula: Quantum Mechanings: A Time-Depentend Perspective by David J. Tannor. Chapter 11)
def gen_xp_trans(x_N,x_gitter,p_gitter):
    a = np.zeros(shape=(len(x_gitter),len(x_gitter)),dtype = "complex_")
    for n in range(len(a)):
        for m in range(len(a)):
            a[n,m] = np.exp((0+1j) * x_gitter[n,n] * p_gitter[m,m]) / np.sqrt(x_N)
    return a


##Generation of the position and kinetic energy operator matrices for the legendre degrees of freedom
##Procedure analogous to harmonic oscillator. Difference being only different matrix entries as equations for rigid rotor (legendre polynomials) are used instead of harmonic oscillator functions
def generate_legendre_dvr(parameter):
    N = parameter["N"]
    x_mat = generate_legendre_x_mat(N)
    x_diag, x_eigen, x_eigen_tr, x_eigenvalues = diagonalize(x_mat)
    l2_mat = generate_legendre_L2_mat(N)
    t_mat = l2_mat / 2
    t_mat_nb = x_eigen_tr @ t_mat @ x_eigen
    return t_mat_nb, x_eigenvalues

def generate_legendre_x_mat(N):    
    a = np.zeros(shape=(N,N))
    for n in range(len(a)):
        if n+1 < len(a):
            a[n,n+1] = (n+1) / np.sqrt((2*n+1) * (2*n+3))           #(Source of the formula: Quantum Mechanings: A Time-Depentend Perspective by David J. Tannor. Chapter 11)
            a[n+1,n] = a[n,n+1]
    return a

def generate_legendre_L2_mat(N):
    a = np.zeros(shape=(N,N))
    for n in range(len(a)):
        a[n,n] = n*(n+1)                                            #Eigenvalues of total angular momentum operator.
    return a


def diagonalize(mat):
    eigenvalues, eigenvectors = np.linalg.eigh(mat)

    idx = eigenvalues.argsort()                                     #Sorting the eigenvalues in ascending order is important for the lanczos algorithm
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    diag = np.zeros(shape=(len(mat),len(mat)),dtype = "complex_")
    for i in range(len(mat)):
        diag[i,i] = eigenvalues[i]

    eigenvectors_tr = np.matrix.transpose(eigenvectors)
    return diag, eigenvectors, eigenvectors_tr, eigenvalues

## main function for the lanczos algorithm performing one lanczos iteration
def lanczos(t_dvr,v,psi,N_lanczos,dims,inertia):
    time_start = time.time()
    krylov_basis = np.zeros(shape=(N_lanczos,len(psi)))
    krylov_basis_var1 = np.zeros(shape=(N_lanczos,len(psi)))

    krylov_basis[0] = psi
    krylov_basis[0] = krylov_basis[0] / np.linalg.norm(krylov_basis[0])

    for n in range(N_lanczos-1):
        krylov_basis_var1[n+1] = h_psi(t_dvr,krylov_basis[n],dims,inertia) + np.multiply(v,krylov_basis[n])#np.matmul(h_mat,krylov_basis[n])   #np.matmul(t_dp,krylov_basis[n]) + np.multiply(v,krylov_basis[n]) #h_psi(t_dvr, krylov_basis[n])
        krylov_basis[n+1] = copy.deepcopy(krylov_basis_var1[n+1])
        for i in range(n+1):
            krylov_basis[n+1] =  krylov_basis[n+1] - np.multiply(krylov_basis[i],np.vdot(krylov_basis[i], krylov_basis[n+1]))
        krylov_basis[n+1] = krylov_basis[n+1] / np.linalg.norm(krylov_basis[n+1])

    h_lanczos = np.zeros(shape=(N_lanczos-1,N_lanczos-1))
    for n in range(N_lanczos-1):
        for m in range(N_lanczos-1):
            if n == m or n == m+1 or n == m-1:
                h_lanczos[n,m] = np.vdot(krylov_basis[n],krylov_basis_var1[m+1])#np.matmul(h_mat,krylov_basis[m]))  #np.matmul(t_dp,krylov_basis[m]) + np.multiply(v,krylov_basis[m])   #h_psi(t_dvr, krylov_basis[m])


    h_lanczos_diag, ev, ev_tr,_ = diagonalize(h_lanczos)

    new_psi = np.zeros(shape=(len(psi)))
    for n in range(N_lanczos-1):
        new_psi += np.multiply(ev[n,0],krylov_basis[n])

    print(f"---{time.time() - time_start}")
    return h_lanczos_diag, new_psi

## Adjusted function from above for the time propagation
def lanczos2(t_dvr,v,psi,N_lanczos,dims,inertia):
    time_start = time.time()
    krylov_basis = np.zeros(shape=(N_lanczos,len(psi)),dtype = "complex_")
    krylov_basis_var1 = np.zeros(shape=(N_lanczos,len(psi)),dtype = "complex_")
    
    krylov_basis[0] = psi
    krylov_basis[0] = krylov_basis[0] / np.linalg.norm(krylov_basis[0])

    for n in range(N_lanczos-1):
        krylov_basis_var1[n+1] = h_psi(t_dvr,krylov_basis[n],dims,inertia) + np.multiply(v,krylov_basis[n])#np.matmul(h_mat,krylov_basis[n])   #np.matmul(t_dp,krylov_basis[n]) + np.multiply(v,krylov_basis[n]) #h_psi(t_dvr, krylov_basis[n])
        krylov_basis[n+1] = copy.deepcopy(krylov_basis_var1[n+1])
        for i in range(n+1):
            krylov_basis[n+1] =  krylov_basis[n+1] - np.multiply(krylov_basis[i],np.vdot(krylov_basis[i], krylov_basis[n+1]))
        krylov_basis[n+1] = krylov_basis[n+1] / np.linalg.norm(krylov_basis[n+1])

    h_lanczos = np.zeros(shape=(N_lanczos-1,N_lanczos-1),dtype = "complex_")
    for n in range(N_lanczos-1):
        for m in range(N_lanczos-1):
            if n == m or n == m+1 or n == m-1:
                h_lanczos[n,m] = np.vdot(krylov_basis[n],krylov_basis_var1[m+1])#np.matmul(h_mat,krylov_basis[m]))  #np.matmul(t_dp,krylov_basis[m]) + np.multiply(v,krylov_basis[m])   #h_psi(t_dvr, krylov_basis[m])

    h_lanczos_diag, ev, ev_tr,_ = diagonalize(h_lanczos)
    new_psi = np.zeros(shape=(len(psi)),dtype = "complex_")
    for n in range(N_lanczos-1):
        new_psi += np.multiply(ev[n,0],krylov_basis[n])
    ev_x = np.zeros(shape=(N_lanczos-1,len(psi)),dtype = "complex_")
    for i in range(N_lanczos-1):
        for n in range(N_lanczos-1):
            ev_x[i] += np.multiply(ev[n,i],krylov_basis[n])
    #print(f"---{time.time() - time_start}")
    return h_lanczos_diag, new_psi, ev, ev_x, krylov_basis
    
## H-Operator Matrix on wavefunction. This is generalized to work for n H-Operator Matrices on an n-D wavefunction
## Due to the lanczos algorithm and time propagation this function is called MANY times throughout the programs runtime. If looking for optimization, improving this function has the biggest impact
def h_psi(h_matrizen,psi,dims,inertia):
    start = time.time()
    new_psi = np.zeros(shape=(len(psi)),dtype = "complex_")
    new_psi2 = np.zeros(shape=(len(psi)),dtype = "complex_")
    tot_dim = len(psi)
    n = len(h_matrizen)
    #temp_psi = np.zeros(shape=(len(psi)))
    #for i in range(len(psi)):
    #    temp_psi[i] = psi[i]
    for i in range(len(h_matrizen)):
        h_matr = h_matrizen[-(i+1)]
        dim = len(h_matr)
        if i == 0:
            past_dim = 1
            future_dim = int(tot_dim / dim)
            testa = np.reshape(psi, [future_dim,dim])
            new_psi = np.reshape(new_psi, [future_dim,dim])
            for j in range(future_dim):
                new_psi[j,:] += np.matmul(h_matr,testa[j,:])
            new_psi = np.reshape(new_psi, tot_dim)
            new_psi = np.multiply(new_psi,inertia)
        elif i == n-1:
            testa = np.reshape(psi, [dim,past_dim])
            #new_psi = np.reshape(new_psi, [dim,past_dim])
            new_psi2 = np.reshape(new_psi2, [dim,past_dim])
            for j in range(past_dim):
                #new_psi[:,j] += np.matmul(h_matr,testa[:,j])
                new_psi2[:,j] += np.matmul(h_matr,testa[:,j])
            new_psi2 = np.reshape(new_psi2, tot_dim)
            new_psi = np.reshape(new_psi, tot_dim)
            #new_psi2 = np.multiply(new_psi2,inertia)
            new_psi += new_psi2
            
        else:
            future_dim = int(tot_dim / (dim*past_dim))
            testa = np.reshape(psi, [future_dim,dim,past_dim])
            new_psi = np.reshape(new_psi, [future_dim,dim,past_dim])
            for j in range(future_dim):
                for k in range(past_dim):
                    new_psi[j,:,k] += np.matmul(h_matr,testa[j,:,k])
            #new_psi = np.reshape(new_psi, tot_dim)
        past_dim *= dim
    #print(time.time() - start)
    #new_psi = np.multiply(new_psi,inertia)
    return new_psi

## Position Operator on wavefunction. This is generalized to work for n Position Operator Matrices on an n-D wavefunction
def x_psi(x_matrizen,psi,dims):
    start = time.time()
    new_psi = np.zeros(shape=(len(psi)),dtype = "complex_")
    new_psi2 = np.zeros(shape=(len(psi)),dtype = "complex_")
    new_psi3 = np.zeros(shape=(len(psi)),dtype = "complex_")
    tot_dim = len(psi)
    n = len(x_matrizen)
    #temp_psi = np.zeros(shape=(len(psi)))
    #for i in range(len(psi)):
    #    temp_psi[i] = psi[i]
    for i in range(len(x_matrizen)):
        h_matr = x_matrizen[-(i+1)]
        h_matr = np.diag(h_matr)
        dim = len(h_matr)
        if i == 0:
            past_dim = 1
            future_dim = int(tot_dim / dim)
            testa = np.reshape(psi, [future_dim,dim])
            new_psi3 = np.reshape(new_psi3, [future_dim,dim])
            for j in range(future_dim):
                new_psi3[j,:] += np.matmul(h_matr,testa[j,:])
            new_psi3 = np.reshape(new_psi3, tot_dim)
            #new_psi = np.multiply(new_psi,inertia)
        elif i == n-1:
            testa = np.reshape(psi, [dim,past_dim])
            new_psi = np.reshape(new_psi, [dim,past_dim])
            for j in range(past_dim):
                new_psi[:,j] += np.matmul(h_matr,testa[:,j])
            new_psi = np.reshape(new_psi, tot_dim)
        else:
            future_dim = int(tot_dim / (dim*past_dim))
            testa = np.reshape(psi, [future_dim,dim,past_dim])
            new_psi2 = np.reshape(new_psi2, [future_dim,dim,past_dim])
            for j in range(future_dim):
                for k in range(past_dim):
                    new_psi2[j,:,k] += np.matmul(h_matr,testa[j,:,k])
            new_psi2 = np.reshape(new_psi2, tot_dim)
        past_dim *= dim
    #print(time.time() - start)
    #new_psi = np.multiply(new_psi,inertia)
    return new_psi,new_psi2,new_psi3

# 1D Potential energy surface of a HO for testing purposes of the 1D case. Not in use
def generate_v(x_p,total_dim):
    #print(type(x_p))
    #print(x_p[0])
    v_temp = np.zeros(shape=(total_dim))
    for i in range(len(v_temp)):
        v_temp[i] = x_p[0][int(i/len(x_p[0]))] ** 2 + x_p[1][i%len(x_p[1])] ** 2
    return v_temp

# 1D Potential energy surface of a HO for testing purposes of the 1D case. Not in use
def get_v(point, w):
    v_temp = 0
    for i in range(len(point)):
        v_temp += 0.5 * w**2 * (point[i] ** 2)
    return v_temp

# Potential energy surface of the ground state
def generate_v_nocl_s0(x_p,total_dim,w):
    v_temp = np.zeros(shape=(total_dim))
    dimensions = [1]
    for i in reversed(range(len(x_p))):
        dimensions.append(len(x_p[i]) * dimensions[-1])
    dimensions = dimensions[::-1]
    print(dimensions)
    for i in range(len(v_temp)):
        point = []
        for j in range(len(x_p)):
            ## get V at the given point
            point.append(x_p[j][int(i/dimensions[j+1]) % len(x_p[j])])
        #print(point)
        v_temp[i] = get_v_nocl_s0(point,w)
    return v_temp

# same functionality as the function above but generalized for n degrees of freedem. Not in use
def generate_v_ndim(x_p,total_dim,w):
    #print(x_p)
    v_temp = np.zeros(shape=(total_dim))
    dimensions = [1]
    for i in reversed(range(len(x_p))):
        dimensions.append(len(x_p[i]) * dimensions[-1])
    dimensions = dimensions[::-1]
    print(dimensions)
    #print(dimensions)
    for i in range(len(v_temp)):
        pointer = []
        for j in range(len(x_p)):
            pointer.append(int(i/dimensions[j+1]) % len(x_p[j]))
        #print(pointer)
        v_temp[i] = get_v(pointer,w)
    return v_temp

## V at a specific point on the PES of the ground state. The data is fitted from empirical data (source: )
def get_v_nocl_s0(point, w):
    #r_v = point[0]         r_v: N-O distance, simulated as HO
    #r_d = point[1]         r_d: N-O-center of mass - Cl distance, simulated by Fourier (dissociative)
    #theta = point[2]       theta: Angle between N-O-line to Cl- NO-center of mass line
    #print(point)
    rel_1 = 16/(16+14)
    rel_2 = 14/(16+14)
    r1 = point[0]
    r2 = np.sqrt(rel_1**2 * point[0]**2 + point[1]**2 + 2*point[0]*point[1]*point[2] * rel_1)
    r3 = np.sqrt(rel_2**2 * point[0]**2 + point[1]**2 - 2*point[0]*point[1]*point[2] * rel_2)

    v1 = 0.5 * 0.8987 * (r1-2.155)**2
    v2 = 0.5 * 0.0874 * (r2-3.729)**2
    v3 = 0.5 * 0.1137 * (r3-4.989)**2
    vr = 0.0122 * (r2-3.729) * (r3-4.989)

    return v1+v2+v3+vr
## Potential energy surface of the excited state
def generate_v_nocl_s1(x_p,total_dim):
    v_temp = np.zeros(shape=(total_dim))
    v_points = np.zeros(shape=(total_dim,4))
    ccc = 0
    dimensions = [1]
    for i in reversed(range(len(x_p))):
        dimensions.append(len(x_p[i]) * dimensions[-1])
    dimensions = dimensions[::-1]
    #print(dimensions)
    for i in range(len(v_temp)):
        point = []
        for j in range(len(x_p)):
            point.append(x_p[j][int(i/dimensions[j+1]) % len(x_p[j])])
        v_temp[i] = get_v_nocl_s1(point)
    #np.savetxt("v1.txt",v_points)
    return v_temp

## V at a specific point on the PES of the excited state. The data is fitted from empirical data (source: )
def get_v_nocl_s1(point):
    if parameter[0]["type"] == "HO":
        r_v = point[0]         #r_v: N-O distance, simulated as HO
        r_d = point[1]         #r_d: N-O-center of mass - Cl distance, simulated by Fourier (dissociative)
    else:
        r_v = point[1]         #r_v: N-O distance, simulated as HO
        r_d = point[0]         #r_d: N-O-center of mass - Cl distance, simulated by Fourier (dissociative)        
    cos_theta = point[2]       #theta: Angle between N-O-line to Cl- NO-center of mass line
    #print(point)
    alpha = 1.5
    beta = 1.1
    rd_e = 4.315
    rv_e = 2.136
    theta_e = 127.4
    cos_theta_e = -0.607375839723287
    a_2 = 0.6816
    a_3 = -0.9123
    a_4 = 0.4115
    
    q_d = 1 - np.exp(-alpha * (r_d - rd_e))
    q_v = r_v - rv_e
    q_theta = np.exp(-beta * cos_theta) - np.exp(-beta * cos_theta_e)

    c = np.zeros(shape=(4,5,7))
    #fitted coefficients
    c[0,0] = [0.0384816,0.0247875,0.0270933,0.00126791,0.00541285,0.0313629,0.0172449]
    c[0,1] = [0.00834237,0.00398713,0.00783319,0.0294887,-0.0154387,-0.0621984,-0.0337951]
    c[0,2] = [0.00161625,-0.00015633,-0.0189982,-0.00753297,0.00383665,-0.00758225,-0.00904493]
    c[0,3] = [-0.0010101,0.000619148,-0.0149812,-0.0199722,0.00873013,0.0376118,0.0221523]
    c[0,4] = [-0.0003689,0.000164037,-0.00331809,-0.00567787,0.00268662,0.0134483,0.0084585]
    c[1,0] = [-0.0558666,-0.0276576,0.0934932,-0.0295638,-0.15436,0.0796119,0.135121]
    c[1,1] = [0.0582169,0.0384404,0.078114,0.185556,-0.0641656,-0.175976,-0.0104994]
    c[1,2] = [0.052285,0.0472724,-0.216008,-0.147775,0.349283,0.28458,0.00384449]
    c[1,3] = [0.0212609,0.0290597,-0.109124,0.0310445,0.262513,-0.250653,-0.369466]
    c[1,4] = [0.00334178,0.0039061,-0.0110452,0.0582029,0.0679524,-0.16459,-0.165337]
    c[2,0] = [-0.163186,-0.180535,0.04692,0.471673,0.403267,-0.718071,-0.761199]
    c[2,1] = [-0.0290674,-0.0136172,-0.108952,-1.68269,-1.2673,3.17648,2.92793]
    c[2,2] = [0.121228,0.202308,0.483613,1.29095,-0.174483,-2.4605,-1.36597]
    c[2,3] = [0.107233,0.115213,-0.366102,0.812662,1.76038,-1.19665,-1.77392]
    c[2,4] = [0.0232767,0.0304932,-0.19455,-0.0307517,0.539365,0.120203,-0.251289]
    c[3,0] = [0.0838975,0.198853,-0.0994766,-0.822409,-0.586006,1.17402,1.17378]
    c[3,1] = [-0.182047,-0.245637,0.130396,2.85439,2.44277,-5.36406,-5.22806]
    c[3,2] = [-0.227493,-0.470604,-0.670555,-1.66997,0.268677,3.71822,2.10678]
    c[3,3] = [-0.13635,-0.193843,0.626076,-1.55192,-3.22512,3.03851,4.01364]
    c[3,4] = [-0.0262554,-0.0391291,0.312858,-0.122063,-1.03112,0.28978,0.878604]

    #print(c[1,3,5])

    v = a_2 * (q_v**2) + a_3 * (q_v**3) + a_4 * (q_v**4)

    for i in range(4):
        for j in range(5):
            for k in range(7):
                v += (1 - q_d) * c[i,j,k] * (q_v**i) * (q_d**j) * (q_theta**k)
    #print(v)
    #print(point,v)
    return v

##Inertia
def generate_I(x_p,total_dim):
    vec = np.zeros(shape=(total_dim))
    n = 0
    red_mass_v = (1/14 + 1/16) / 1836           #reduced mass of NO
    red_mass_d = (1/(14+16) + 1/35) / 1836      #reduced mass of point of mass of (NO) and Cl
    for i in range(len(x_p[0])):
        for j in range(len(x_p[1])):
            for k in range(len(x_p[2])):
                vec[n] = red_mass_v * 1/(x_p[0][i])**2 + red_mass_d * 1/(x_p[1][j])**2
                n += 1
    return vec


def main():  
    tx = time.time()
    t_dvr = []
    x_p = []
    total_dim = 1
    dims = []
    my_v = (1/14 + 1/16) / 1836                                 #reduced mass of NO
    my_d = (1/(14+16) + 1/35) / 1836                            #reduced mass of point of mass of (NO) and Cl

    ## Generierung der Matrixdarstellungen der kinetischen Energieoperators sowie des Ortsoperators aller Freiheitsgrade im jeweils ausgewählten Modell (type).
    for i in range(len(parameter)):
        if parameter[i]["type"] == "HO":
            t_dvr_s,x_p_s = generate_ho_dvr(parameter[i])
            x_p_s = x_p_s / np.sqrt(parameter[i]["w"])
            x_p_s = x_p_s / np.sqrt(1/my_v)
            x_p_s += parameter[i]["shift"]
            t_dvr_s = t_dvr_s * parameter[i]["w"]
            t_dvr.append(-t_dvr_s)
            x_p.append(x_p_s)
            total_dim *= parameter[i]["N"]
            dims.append(parameter[i]["N"])
            w = parameter[i]["w"]
        elif parameter[i]["type"] == "Fourier":
            t_dvr_s,x_p_s = generate_fourier_dvr(parameter[i])
            t_dvr_s = t_dvr_s * my_d
            t_dvr.append(t_dvr_s)
            x_p.append(x_p_s)
            total_dim *= parameter[i]["xN"]
            dims.append(parameter[i]["xN"])
        elif parameter[i]["type"] == "Legendre":
            t_dvr_s,x_p_s = generate_legendre_dvr(parameter[i])
            t_dvr.append(t_dvr_s)
            x_p.append(x_p_s)
            total_dim *= parameter[i]["N"]
            dims.append(parameter[i]["N"])

    print(f"Chosen wavefunction dimensions: {dims}")

    inertia = generate_I(x_p,total_dim)
    
    psi = generate_psi(total_dim)   ### Initial guess of the wavefunction

    v0 = generate_v_nocl_s0(x_p,total_dim, w)
    v1 = generate_v_nocl_s1(x_p,total_dim)

    ##Lanczos algorithm is used to avoid the absurd computational effort of diagonalizing a 10^5 x 10^5 hamilton matrix. Still very costly though
    h_lanczos, new_psi = lanczos(t_dvr,v0,psi,N_lanczos,dims,inertia)
    print(h_lanczos[0,0])
    prev_ev = 0
    while abs(h_lanczos[0,0] - prev_ev) > 10**(-8):
        prev_ev = h_lanczos[0,0]
        h_lanczos, new_psi = lanczos(t_dvr,v0,new_psi,N_lanczos,dims,inertia)
        print(h_lanczos[0,0])

    print("Ground state wave function converged")
    print(f"It took:   {time.time() - tx} seconds.")
    #np.savetxt("wfsave.txt", new_psi)
    #new_psi = np.loadtxt("wfsave.txt")
    #print(np.vdot(new_psi,psi_read))
    
    ### P2
    print("Starting time propagation of the dissociation process. This will take a while (hours)")
    psi0 = copy.deepcopy(new_psi)
    psi_curr = copy.deepcopy(psi0)
    #print(np.linalg.norm(psi_curr))

    tottime = 0
    tstep2 = 0.1
    
    overlap00 = [[1,1,1,1],[1,1,1,1]]
    overlap000 = [1]
    en_ew = np.array([1])
    x_ew = [[1,1,1],[1,1,1]]
    x_ew1 = np.array([1])
    x_ew2 = np.array([1])
    x_ew3 = np.array([1])
    smallerstep = False

    while tottime < 1984:   #48 fs
        tstep_to_full_number = int(tottime + 1) - tottime
        tstep = min(tstep2,tstep_to_full_number)
        print("--")
        ## Forward Propagation: Calculating Wavefunction at psi_t1=psi_(t0+Delta_t)
        ## smallerstep == False means the previous propagation step was successfull and we have to calculate psi_t0 and then psi_t1
        if smallerstep == False:
            psi_lcz2 = copy.deepcopy(psi_curr)
            h_lanczos,_,ev,ev_xbasis,krylov_basis = lanczos2(t_dvr,v1,psi_lcz2,11,dims,inertia)
            psi_next = np.zeros(shape=(len(psi)),dtype = "complex_")
            for i in range(10):
                ovl = np.vdot(ev_xbasis[i],psi_curr)
                psi_next += np.exp((0-1j) * h_lanczos[i,i] * tstep) * ev_xbasis[i] * ovl
        ## smallerstep == False means the previous propagation step was too big. We therefore start with the same psi_t0 and only need to calculate psi_t1
        else:
            psi_next = np.zeros(shape=(len(psi)),dtype = "complex_")
            for i in range(10):
                ovl = np.vdot(ev_xbasis[i],psi_curr)
                psi_next += np.exp((0-1j) * h_lanczos[i,i] * tstep) * ev_xbasis[i] * ovl

        ###back-propagation: Used to check backwards if the timestep was "small enough"
        psi_lczz2 = copy.deepcopy(psi_next)
        h_lanczos_b,_,ev_b,ev_xbasis_b,krylov_basis_b = lanczos2(t_dvr,v1,psi_lczz2,11,dims,inertia)
        psi_back = np.zeros(shape=(len(psi)),dtype = "complex_")
        for i in range(10):
            ovl_b = np.vdot(ev_xbasis_b[i],psi_next)
            psi_back += np.exp((0+1j) * h_lanczos_b[i,i] * (tstep)) * ev_xbasis_b[i] * ovl_b
        #psi_back = psi_back / np.linalg.norm(psi_back)

        ## Check overlap of psi_t0 with psi_(t1-delta_t). If they deviate significantly, the timestep was too big.
        ovvv = 0
        for i in range(len(psi_curr)):
            ovvv += (psi_curr[i] - psi_back[i])**2
        ovvv = ovvv / len(psi_curr)
        ovvv = np.absolute(ovvv)
        print(f"Overlap: {ovvv}")
        ## Update the parameters and wavefunctions if the timestep was accepted. 
        if ovvv < 10**(-24):
            psi_curr = copy.deepcopy(psi_next)
            tottime += tstep
            print(f"Tstep: {round(tstep,4)},Ttot: {round(tottime,4)}")
            if round(tstep,3) == round(tstep2,3):                               #just so the tstep isnt increased in case tstep_to_full_number was used
                tstep2 *= 1.1                                                   #make the timesteps slightly bigger so the calculation takes less steps. Will get corrected down automatically once they become too big
            tstep2 = min(tstep2,1)                                              #1 is max as we want the numbers for every atomic unit of time
            overlap0 = np.vdot(psi_next,psi0)
            if round(tottime,7) + 0.0000001  >= len(overlap00)-1:
                overlap00.append([np.real(overlap0),np.imag(overlap0),np.absolute(overlap0)])
                overlap000.append(np.absolute(overlap0))
                print(f"Recorded overlap: {overlap00[-1]}")
            smallerstep = False
            
            ## commented code to calculate expectation values for the position operator. Basically a "live" view of the dissociation process. 
##                h_psii = h_psi(t_dvr,psi_next,dims,inertia) + np.multiply(v1,psi_next)
##                en_ew = np.append(en_ew,np.real(np.vdot(psi_next,h_psii)))
##                print(f"Energieerwartungswert: {en_ew[-1]}")
##
##                x_psii1,x_psii2,x_psii3 = x_psi(x_p,psi_next,dims)
##                x_ew1 = np.real(np.vdot(psi_next,x_psii1))
##                x_ew2 = np.real(np.vdot(psi_next,x_psii2))
##                x_ew3 = np.real(np.vdot(psi_next,x_psii3))
##                x_ew.append([x_ew1,x_ew2,x_ew3])
##                print(f"Ortserwartungswert: {x_ew[-1]}")
 
        else:
            print(f"Step too big: {tstep}")
            tstep2 *= 0.8                                   ## make timestep smaller and repeat calculation
            smallerstep = True
##        if tottime + tstep > 1983.45:
##            tstep = 1983.47 - tottime
##            smallerstep = False
    print(time.time() - tx)
    np.savetxt("ovl.txt", overlap00[2:])
    np.savetxt("ovlabs.txt", overlap000[1:])
    np.savetxt("wf_final.txt",psi_next)
    #np.savetxt("wf_final_sanity.txt",psi_curr)
    exit()


if __name__ == "__main__":
    main()