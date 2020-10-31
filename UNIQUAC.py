import numpy as np                           # Matrizes
import math as mt                            # Ferramentas Matemáticas
from matplotlib.pyplot import plot,scatter, show              # Plotagem de Gráfico


P = 95458.552                                # Pressão do Sistema
R = 8.3144621                                # Constante Universal dos Gases Perfeitos
Tc = np.array([[553.50], [591.80]])          # Temperatura Crítica ( Ciclo-hexano / Tolueno )
Pc = np.array([[4070000], [4100000]])        # Pressão Crítica ( Ciclo-hexano / Tolueno )
Vc = np.array([[0.000308], [0.000316]])      # Volume Crítico ( Ciclo-hexano / Tolueno )
W = np.array([[0.212], [0.263]])             # Fator Acêntrico ( Ciclo-hexano / Tolueno )
Z = np.array([[0.2729], [0.2644]])           # Z de Rackett ( Ciclo-hexano / Tolueno )
A = np.array([[116.51], [80.877]])           # DIPPR para cálculo de Psat
B = np.array([[-7103.3], [-6902.4]])         # DIPPR para cálculo de Psat
C = np.array([[-15.490], [-8.7761]])         # DIPPR para cálculo de Psat
D = np.array([[0.016959], [0.0000058034]])   # DIPPR para cálculo de Psat
E = np.array([[1], [2]])                     # DIPPR para cálculo de Psat
r = np.array([[4.0464], [3.9228]])
q = np.array([[3.240], [2.968]])
z = 10
I = np.array([[z*(r[0]-q[0])/2 - (r[0]-1)], [z*(r[1]-q[1])/2 - (r[1] - 1)]])
Vc12 = ((Vc[0] ** (1 / 3) + Vc[1] ** (1 / 3)) / 2) ** 3

Tc12 = ((Tc[0] * Tc[1]) ** 0.5)
Pc12 = 0.2685 * R * Tc12 / (Vc12)

w12 = (W[0] + W[1]) / 2

def psat(T): # calculo da pressão de saturaçãoo
    psat1 = mt.exp(A[0] + B[0]/T + C[0]*mt.log(T) + D[0]*T**E[0])
    psat2 = mt.exp(A[1] + B[1]/T + C[1]*mt.log(T) + D[1]*T**E[1])
    return np.array([[psat1],[psat2]])

def Temp(x1,T):  ## Estimativa inicial pela Lei de Raul da temperatura
    return  -P + x1*psat(T)[0] + (1-x1)*psat(T)[1]

def dTemp(x1,T):
    h = 1e-6
    return (Temp(x1,T+h)-Temp(x1,T-h))/(2*h)

def NR(x1,T):
    B = 1
    while abs(Temp(x1,T))>1e-8 :
            t1 = T-B*Temp(x1,T)/dTemp(x1,T)
            if abs(Temp(x1,t1)) < abs(Temp(x1,T)):
                T = t1
                B = 1
            else: B = B*0.5
    return T

def t(T):    # Cálculo do Tau
    return np.array([[mt.exp(-1*((320.37+ 0.1048*(T-298.15)))/T)] , [mt.exp(-1*((-203.21 + 0.1168*(T-298.15)))/T)]])

def fi(x1):   # Cáculo do Φ
    fi1 = x1 * r[0] / (x1 * r[0] + (1 - x1) * r[1])
    fi2 = 1 - fi1
    return np.array([[fi1],[fi2]])

def teta(x1):   # Cáculo do teta
    tt1 = x1 * q[0] / (x1 * q[0] + (1 - x1) * q[1])
    tt2 = 1 - tt1
    return np.array([[tt1],[tt2]])

def gama(x1,T):
    gm1 = mt.exp( mt.log(fi(x1)[0] / x1) + 5 * q[0] * mt.log(teta(x1)[0] / fi(x1)[0]) + I[0] - fi(x1)[0] * (x1 * I[0] + (1 - x1) *I[1]) / x1 - q[0] * mt.log(teta(x1)[0] + teta(x1)[1]*t(T)[1]) + q[0] - q[0] * (teta(x1)[0] / (teta(x1)[0] + teta(x1)[1] * t(T)[1] ) + teta(x1)[1] * t(T)[0] / (teta(x1)[0] * t(T)[0]  + teta(x1)[1])))
    gm2 = mt.exp(mt.log(fi(x1)[1] / (1-x1)) + 5 * q[1] * mt.log(teta(x1)[1] / fi(x1)[1]) + I[1] - fi(x1)[1] * (x1 * I[0] + (1 - x1) * I[1]) / (1-x1) - q[1] * mt.log(teta(x1)[0]*t(T)[0] + teta(x1)[1]) + q[1] - q[1] * (teta(x1)[0]*t(T)[1] / (teta(x1)[0] + teta(x1)[1]*t(T)[1]) + teta(x1)[1] / (teta(x1)[0] * t(T)[0]  + teta(x1)[1] )))
    return np.array([[gm1],[gm2]])

def VL (T):
    VL1 = (R * Tc[0] * Z[0] ** (1 + (1 - T/Tc[0] ) ** (2 / 7))) / (Pc[0])
    VL2 = (R * Tc[1] * Z[1] ** (1 + (1 - T /Tc[1]) ** (2 / 7))) / (Pc[1])
    return np.array([[VL1], [VL2]])

def POY(T):
    POY1 = mt.exp((P - psat(T)[0]) * VL(T)[0] / (R* T))
    POY2 = mt.exp((P - psat(T)[1]) * VL(T)[1] / (R * T))
    return np.array([[POY1], [POY2]])

def Bi(T):
    B11 = R * Tc[0] * (0.083 - 0.422 / (T/Tc[0]) ** 1.6 + W[0] * (0.139 - 0.172 / (T/ Tc[0]) ** 4.2)) / Pc[0]
    B22 = R * Tc[1] * (0.083 - 0.422 / (T / Tc[1]) ** 1.6 + W[1] * (0.139 - 0.172 / ((T ) / Tc[1]) ** 4.2)) / Pc[1]
    B12 = R * Tc12 * (0.083 - 0.422 / (T/Tc12) ** 1.6 + w12 * (0.139 - 0.172 / (T / Tc12) ** 4.2)) / Pc12
    return np.array([[B11], [B22], [B12]])

def FIV(y1,T):
    FIV1 = mt.exp(P * (2 * (y1 * Bi(T)[0] + (1 - y1) * Bi(T)[2]) - (Bi(T)[0] * y1 ** 2 + 2 * Bi(T)[2] * y1 * (1 - y1) + Bi(T)[1] * (1 - y1) ** 2)) / (R * T))
    FIV2 = mt.exp(P * (2 * (y1 * Bi(T)[2] + (1 - y1) * Bi(T)[1]) - y1 ** 2 * Bi(T)[0] - 2 * y1 * (1 - y1) * Bi(T)[2] - (1 - y1) ** 2 * Bi(T)[1]) / (R * T))
    return np.array([[FIV1], [FIV2]])

def FIsat(T):
    FIsat1 = mt.exp(Bi(T)[0] * psat(T)[0] / (R * (T)))
    FIsat2 = mt.exp(Bi(T)[1] * psat(T)[1] / (R * (T )))
    return np.array([[FIsat1], [FIsat2]])

def K11(x1,y,T):
    f = x1*gama(x1, T)[0] * psat(T)[0] * FIsat(T)[0] * POY(T)[0] / (P * FIV(y, T)[0]) +(1-x1)*gama(x1,T)[1] * psat(T)[1] * FIsat(T)[1] * POY(T)[1] / (P * FIV(y, T)[1]) -1
    return f

def dK11(x1,y,T):
    h = 1e-6
    return (K11(x1,y,T+h)-K11(x1,y,T-h))/(2*h)

def NK11(x1,y,T):
    B = 1
    while abs(K11(x1,y,T))>1e-3 :
            t1 = T-B*K11(x1,y,T)/dK11(x1,y,T)
            if abs(K11(x1,y,t1)) < abs(K11(x1,y,T)):
                T = t1
                B = 1
            else: B = B*0.5
    return T

def f(x1):
    T = T1 = NR(x1,300)  # temperatura de chute ( pensar bem ) # se for lei de raul, usar as médias das temperaturas de saturação obtidas analíticamentes
    y = x1*psat(T)[0]/P
    while True:
        y1 = y
        T=T1
        K11 = gama(x1, T)[0] * psat(T)[0] * FIsat(T)[0] * POY(T)[0] / (P * FIV(y1, T)[0])
        K22 = gama(x1, T)[1] * psat(T)[1] * FIsat(T)[1] * POY(T)[1] / (P * FIV(y1, T)[1])
        y = K11 * x1 / (K11 * x1 + K22 * (1 - x1))
        if abs(y1 - y) < 1e-3:
            T1 = NK11(x1, y, T)
            if abs(K11 * x1 + K22 * (1 - x1) - 1) < 1e-3:
                return y1, T1
                break





x10=list()
y10 = list()
t10 = list()

x11=(0, 0.043, 0.1, 0.122, 0.272, 0.38, 0.437, 0.566, 0.609, 0.687, 0.795, 0.823, 0.852, 0.895, 0.902, 1)
y11=(0 ,0.080, 0.236, 0.272, 0.501, 0.616, 0.666, 0.75, 0.788, 0.845, 0.895, 0.902, 0.923, 0.938, 0.945, 1)
t11=(107.90, 106.10, 102.40, 101.50, 95.90, 91.90, 90, 86.60, 85.60, 83.70, 81.70, 81.40, 80.80, 80.20, 80, 78.25)


for i in range(1,200):
    x10.append(i/200)
    y10.append(f(i/200)[0][0])
    t10.append(f(i/200)[1][0]-273.15)



plot( x10 ,t10 , color='green' )
plot(y10,t10, color = 'red')
scatter( x11 ,t11 , color='black' )
scatter(y11,t11 , color = 'orange')
show()