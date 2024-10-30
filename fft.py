import numpy as np

filename = "ovl.txt"
omega_start = 0.6
omega_end = 2.0
omega_steps = 141

def main():
    data = np.loadtxt(filename)
    data = data[5:]

    data = data[0::5]

    data_real = data[0:,0]
    data_imag = data[0:,1]

    omegas = []
    omega_delta = (omega_end - omega_start) / (omega_steps-1)
    for x in range(omega_steps):
        omegas.append((omega_start + x*omega_delta) / 27.2114)
    #print(omegas)
    #omega = 0.8 / 27.2114
    
    intensities = []
    
    for s in range(len(omegas)):
        intensity = 1 * np.exp(1j * omegas[s] * 1 * 0)
        for x in range(len(data)):
            overlap = data_real[x]+data_imag[x]*1j
            intensity += overlap.conj() * np.exp(1j * omegas[s] * 1 * (x+1))
        intensity = omegas[s] * np.real(intensity)
        intensities.append(intensity)
        #print(intensity)
    #print(intenisities[20])
    
    omegas = []
    for x in range(omega_steps):
        omegas.append(omega_start + x*(omega_end-omega_start)/(omega_steps-1))
    np.savetxt("spectrum.txt",np.c_[omegas,intensities])

if __name__ == "__main__":
    main()
