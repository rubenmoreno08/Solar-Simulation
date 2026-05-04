# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:29:02 2025

@author: Joel,Ruben,Xavi,Arnau
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import plotly.graph_objects as go
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time



# Constants
G = 6.67430e-11  # Constant gravitacional (m^3 kg^-1 s^-2)
M_sol = 1.989e30  # Massa del Sol (kg)
m = 5.97e24    # Massa de la terra (kg)
UA = 1.496e11  # Unitat Astronòmica (m)
area = 2  # Àrea del panell solar en m^2

# Unitats normalitzades
l0 = UA
t0 = np.sqrt(l0**3 / (G * M_sol))
v0 = l0 / t0

# Posició geogràfica de Barcelona
lat_barcelona = 41.3851 * np.pi / 180  # Latitud en radians
lon_barcelona = 2.1734 * np.pi / 180  # Longitud en radians
alt_barcelona = 12 / UA  # Altitud normalitzada en UA

# Inclinació de l'eix terrestre
epsilon = 23.44 * np.pi / 180  # Inclinació en radians

# Condicions inicials
x0 = -0.98329  # Posició inicial en x (UA)
y0 = 0.0  # Posició inicial en y (UA)
vx0 = 0.0  # Velocitat inicial en x (normalitzada)
vy0 = 30.291e3 / v0  # Velocitat inicial en y (normalitzada)

desfasament_periheli = np.radians(282.9 - 180)  # Desfasament en radians
x0 = -0.98329 * np.cos(desfasament_periheli)
y0 = -0.98329 * np.sin(desfasament_periheli)
vx0 = vy0 * np.sin(desfasament_periheli)
vy0 = -vy0 * np.cos(desfasament_periheli)

# Inicialització de variables
x_euler, y_euler, vx_euler, vy_euler = x0, y0, vx0, vy0
x_rk4, y_rk4, vx_rk4, vy_rk4 = x0, y0, vx0, vy0

# Llistes per emmagatzemar les solucions
x_euler_llista, y_euler_llista = [x_euler], [y_euler]
vx_euler_llista, vy_euler_llista = [vx_euler], [vy_euler]
x_rk4_llista, y_rk4_llista = [x_rk4], [y_rk4]
vx_rk4_llista, vy_rk4_llista = [vx_rk4], [vy_rk4]

# Funcions auxiliars
def acceleracions(x, y):
    r = np.sqrt(x**2 + y**2)
    ax = -x / r**3
    ay = -y / r**3
    return ax, ay

def geodetic_a_ecef(lat, lon, alt):
    a = 6378137.0 / UA  # Semieix major (UA)
    e2 = 0.00669437999014  # Excentricitat quadrada
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([X, Y, Z])

def ecef_a_enu(x_ecef, y_ecef, z_ecef, lat_ref, lon_ref, alt_ref):
    ref_ecef = geodetic_a_ecef(lat_ref, lon_ref, alt_ref)
    dx = x_ecef - ref_ecef[0]
    dy = y_ecef - ref_ecef[1]
    dz = z_ecef - ref_ecef[2]

    R = np.array([
        [-np.sin(lon_ref), np.cos(lon_ref), 0],
        [-np.sin(lat_ref) * np.cos(lon_ref), -np.sin(lat_ref) * np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref) * np.cos(lon_ref), np.cos(lat_ref) * np.sin(lon_ref), np.sin(lat_ref)]
    ])

    enu = R @ np.array([dx, dy, dz])
    return enu

# Paràmetres comuns
N_passos = 8760  # Nombre de passos d'integració (hores en un any)
dt = 2 * np.pi / N_passos  # Pas de temps en anys fraccionats

# Mètodes d'Euler i RK4 en paral·lel
x_rot_euler, y_rot_euler, z_rot_euler = [], [], []
x_rot_rk4, y_rot_rk4, z_rot_rk4 = [], [], []

for i in range(N_passos-1):
    # Euler
    ax_euler, ay_euler = acceleracions(x_euler, y_euler)
    vx_euler += ax_euler * dt
    vy_euler += ay_euler * dt
    x_euler += vx_euler * dt
    y_euler += vy_euler * dt

    x_euler_llista.append(x_euler)
    y_euler_llista.append(y_euler)
    vx_euler_llista.append(vx_euler)
    vy_euler_llista.append(vy_euler)
    # RK4
    ax1, ay1 = acceleracions(x_rk4, y_rk4)
    k1_vx, k1_vy = ax1 * dt, ay1 * dt
    k1_x, k1_y = vx_rk4 * dt, vy_rk4 * dt

    ax2, ay2 = acceleracions(x_rk4 + 0.5 * k1_x, y_rk4 + 0.5 * k1_y)
    k2_vx, k2_vy = ax2 * dt, ay2 * dt
    k2_x, k2_y = (vx_rk4 + 0.5 * k1_vx) * dt, (vy_rk4 + 0.5 * k1_vy) * dt

    ax3, ay3 = acceleracions(x_rk4 + 0.5 * k2_x, y_rk4 + 0.5 * k2_y)
    k3_vx, k3_vy = ax3 * dt, ay3 * dt
    k3_x, k3_y = (vx_rk4 + 0.5 * k2_vx) * dt, (vy_rk4 + 0.5 * k2_vy) * dt

    ax4, ay4 = acceleracions(x_rk4 + k3_x, y_rk4 + k3_y)
    k4_vx, k4_vy = ax4 * dt, ay4 * dt
    k4_x, k4_y = (vx_rk4 + k3_vx) * dt, (vy_rk4 + k3_vy) * dt

    vx_rk4 += (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
    vy_rk4 += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
    x_rk4 += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    y_rk4 += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6

    x_rk4_llista.append(x_rk4)
    y_rk4_llista.append(y_rk4)
    vx_rk4_llista.append(vx_rk4)
    vy_rk4_llista.append(vy_rk4)
    # Rotació de coordenades al sistema equatorial
    z = 0
    matriu_rotacio = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon), -np.sin(epsilon)],
        [0, np.sin(epsilon), np.cos(epsilon)]
    ])
    rotat_rk4 = matriu_rotacio @ np.array([x_rk4, y_rk4, z])
    x_rot_rk4.append(rotat_rk4[0])
    y_rot_rk4.append(rotat_rk4[1])
    z_rot_rk4.append(rotat_rk4[2])
    rotat_euler = matriu_rotacio @ np.array([x_euler, y_euler, z])
    x_rot_euler.append(rotat_euler[0])
    y_rot_euler.append(rotat_euler[1])
    z_rot_euler.append(rotat_euler[2])

# Incloure la rotació diària de la Terra
e_llista_euler, n_llista_euler, u_llista_euler, altituds_euler = [], [], [], []
e_llista_rk4, n_llista_rk4, u_llista_rk4, altituds_rk4 = [], [], [], []
omega_terra = 2 * np.pi / 24

for k in range(24 * 365):
    theta = omega_terra * k
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rotat_euler = R_z @ np.array([x_rot_euler[k % len(x_rot_euler)], y_rot_euler[k % len(y_rot_euler)], z_rot_euler[k % len(z_rot_euler)]])
    enu_euler = ecef_a_enu(rotat_euler[0], rotat_euler[1], rotat_euler[2], lat_barcelona, lon_barcelona, alt_barcelona)
    e_llista_euler.append(enu_euler[0])
    n_llista_euler.append(enu_euler[1])
    u_llista_euler.append(enu_euler[2])
    r_horizontal_euler = np.sqrt(enu_euler[0]**2 + enu_euler[1]**2)
    altitud_euler = np.arctan2(enu_euler[2], r_horizontal_euler) * 180 / np.pi
    altituds_euler.append(altitud_euler)

    rotat_rk4 = R_z @ np.array([x_rot_rk4[k % len(x_rot_rk4)], y_rot_rk4[k % len(y_rot_rk4)], z_rot_rk4[k % len(z_rot_rk4)]])
    enu_rk4 = ecef_a_enu(rotat_rk4[0], rotat_rk4[1], rotat_rk4[2], lat_barcelona, lon_barcelona, alt_barcelona)
    e_llista_rk4.append(enu_rk4[0])
    n_llista_rk4.append(enu_rk4[1])
    u_llista_rk4.append(enu_rk4[2])
    r_horizontal_rk4 = np.sqrt(enu_rk4[0]**2 + enu_rk4[1]**2)
    altitud_rk4 = np.arctan2(enu_rk4[2], r_horizontal_rk4) * 180 / np.pi
    altituds_rk4.append(altitud_rk4)


# Paràmetres del panell solar
luminositat_solar = 3.828e26
energia_per_hora_euler, energia_per_dia_euler, energia_acumulada_euler = [], [], 0
energia_per_hora_rk4, energia_per_dia_rk4, energia_acumulada_rk4 = [], [], 0

Dies = np.arange(1, 366)

for i, altitud in enumerate(altituds_rk4):
    if altitud > 0:
        cos_theta = np.cos(np.radians(90 - altitud))
        irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
        potencia_rebuda = area * cos_theta * irradiancia * 1e-3
        potencia_utilitzada = 0.4 * potencia_rebuda
        energia_rk4 = potencia_utilitzada
    else:
        energia_rk4 = 0
    energia_per_hora_rk4.append(energia_rk4)
    if (i + 1) % 24 == 0:
        energia_total_dia_rk4 = sum(energia_per_hora_rk4[-24:])
        energia_per_dia_rk4.append(energia_total_dia_rk4)
        energia_acumulada_rk4 += energia_total_dia_rk4

for i, altitud in enumerate(altituds_euler):
    if altitud > 0:
        cos_theta = np.cos(np.radians(90 - altitud))
        irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
        potencia_rebuda = area * cos_theta * irradiancia * 1e-3
        potencia_utilitzada = 0.4 * potencia_rebuda
        energia_euler = potencia_utilitzada
    else:
        energia_euler = 0
    energia_per_hora_euler.append(energia_euler)
    if (i + 1) % 24 == 0:
        energia_total_dia_euler = sum(energia_per_hora_euler[-24:])
        energia_per_dia_euler.append(energia_total_dia_euler)
        energia_acumulada_euler += energia_total_dia_euler

# Resultats
print(f"Energia total generada en un any (RK4): {energia_acumulada_rk4:.2f} kWh\n")
print(f"Energia total generada en un any (Euler): {energia_acumulada_euler:.2f} kWh\n")


# Gràfics dels resultats
# Gràfic 1: Òrbites calculades amb Euler i RK4
plt.figure(figsize=(10, 10))
plt.plot(x_euler_llista, y_euler_llista, label="\u00d2rbita - Euler", linestyle="--")
plt.plot(x_rk4_llista, y_rk4_llista, label="\u00d2rbita - RK4")
plt.plot(x_euler_llista[0], y_euler_llista[0], 'o', color="g", label="Periheli")
plt.scatter(0, 0, color="red", label="Sol")
plt.xlabel("x (UA)")
plt.ylabel("y (UA)")
plt.axis("equal")
plt.legend()
plt.title("Comparació d'\u00f2rbites calculades amb Euler i RK4")
plt.grid()
plt.savefig("orbites_euler_vs_rk4.png")  # Guardar el gràfic com a PNG
plt.close()
# Gràfic 1.1 i 1.2: Errors a partir de les energies de l'òrbita



# Funció per calcular les energies
def calcular_energies(x, y, vx, vy):
    r = np.sqrt(x**2 + y**2)  # Distància a l'origen (radi)
    v2 = (vx**2 + vy**2)  # Velocitat al quadrat (assumeix v = (dx/dt, dy/dt))
    energia_cinetica = 0.5 * m * v2 / M_sol
    energia_potencial = - m / (M_sol * r)
    return energia_cinetica, energia_potencial

# Calcula les energies per a Euler
energies_euler = [calcular_energies(x, y, vx, vy) for x, y, vx, vy in zip(x_euler_llista, y_euler_llista, vx_euler_llista, vy_euler_llista)]
energia_total_euler = [sum(e) for e in energies_euler]

# Calcula les energies per a RK4
energies_rk4 = [calcular_energies(x, y, vx, vy) for x, y, vx, vy in zip(x_rk4_llista, y_rk4_llista, vx_rk4_llista, vy_rk4_llista)]
energia_total_rk4 = [sum(e) for e in energies_rk4]

# Energia inicial per a cada mètode
energia_inicial_euler = energia_total_euler[0]
energia_inicial_rk4 = energia_total_rk4[0]

# Càlcul de les desviacions percentuals per a cada punt
desviacions_percentuals_euler = [
    abs((energia - energia_inicial_euler) / energia_inicial_euler) * 100
    for energia in energia_total_euler
]
desviacions_percentuals_rk4 = [
    abs((energia - energia_inicial_rk4) / energia_inicial_rk4) * 100
    for energia in energia_total_rk4
]

# Gràfic de les energies totals
plt.figure(figsize=(10, 6))
plt.plot(range(len(energia_total_euler)), energia_total_euler, label='Energia Total Euler')
plt.plot(range(len(energia_total_rk4)), energia_total_rk4, label='Energia Total RK4')
plt.title('Energia Total de l’òrbita en cada punt')
plt.xlabel('Punt de l’òrbita')
plt.ylabel('Energia total')
plt.legend()
plt.grid()
plt.savefig("energia_total.png")  # Guardar el gràfic com a fitxer PNG
plt.close()

# Gràfic de les desviacions percentuals
plt.figure(figsize=(10, 6))
plt.plot(range(len(desviacions_percentuals_euler)), desviacions_percentuals_euler, label='Desviació Euler')
plt.plot(range(len(desviacions_percentuals_rk4)), desviacions_percentuals_rk4, label='Desviació RK4')
plt.title('Desviació percentual de l’energia')
plt.xlabel('Punt de l’òrbita')
plt.ylabel('Desviació percentual (%)')
plt.legend()
plt.grid()
plt.savefig("desviacio_percentual.png")  # Guardar el gràfic com a fitxer PNG
plt.close()

# Gràfic 2: Energia generada durant l'any (simulació)
plt.figure(figsize=(12, 6))
plt.plot(Dies, energia_per_dia_rk4, label="Energia diària (RK4)")
plt.plot(Dies, energia_per_dia_euler, label="Energia diària (Euler)")
plt.xlabel("Dia de l'any")
plt.ylabel("Energia generada (kWh)")
plt.title("Energia generada pel panell solar durant un any (simulació)")
plt.grid()
plt.legend()
plt.savefig("energia_simulacio.png")  # Guardar el gràfic com a fitxer PNG
plt.close()

declinacions_euler, equacio_del_temps_euler = [], []
declinacions_rk4, equacio_del_temps_rk4 = [], []

for dia in range(365):
    k = dia * 24
    r_total_euler = np.sqrt(x_rot_euler[k]**2 + y_rot_euler[k]**2 + z_rot_euler[k]**2)
    declinacio_euler = np.arcsin(z_rot_euler[k] / r_total_euler) * 180 / np.pi
    declinacions_euler.append(declinacio_euler)
    longitud_solar_euler = np.arctan2(y_rot_euler[k], x_rot_euler[k]) * 180 / np.pi
    longitud_solar_euler = (longitud_solar_euler + 360) % 360
    temps_solar_mitja_euler = (360 / 365.25) * dia
    E_euler = longitud_solar_euler - temps_solar_mitja_euler
    if E_euler > 180:
        E_euler -= 360
    elif E_euler < -180:
        E_euler += 360
    equacio_del_temps_euler.append(E_euler)

for dia in range(365):
    k = dia * 24
    r_total_rk4 = np.sqrt(x_rot_rk4[k]**2 + y_rot_rk4[k]**2 + z_rot_rk4[k]**2)
    declinacio_rk4 = np.arcsin(z_rot_rk4[k] / r_total_rk4) * 180 / np.pi
    declinacions_rk4.append(declinacio_rk4)
    longitud_solar_rk4 = np.arctan2(y_rot_rk4[k], x_rot_rk4[k]) * 180 / np.pi
    longitud_solar_rk4 = (longitud_solar_rk4 + 360) % 360
    temps_solar_mitja_rk4 = (360 / 365.25) * dia
    E_rk4 = longitud_solar_rk4 - temps_solar_mitja_rk4
    if E_rk4 > 180:
        E_rk4 -= 360
    elif E_rk4 < -180:
        E_rk4 += 360
    equacio_del_temps_rk4.append(E_rk4)

equacio_del_temps_min_euler = [e * 4 for e in equacio_del_temps_euler]
equacio_del_temps_min_rk4 = [e * 4 for e in equacio_del_temps_rk4]

# Gràfic: Analema solar observat des de Barcelona
plt.figure(figsize=(10, 6))
plt.plot(equacio_del_temps_min_euler, declinacions_euler, label="Analema (Euler)", color="blue")
plt.plot(equacio_del_temps_min_rk4, declinacions_rk4, label="Analema (RK4)", color="red")
plt.xlabel("Equació del temps (minuts)")
plt.ylabel("Declinació solar (°)")
plt.title("Analema solar observat des de Barcelona")
plt.grid()
plt.legend()
plt.savefig("analema_solar_barcelona.png")  # Guardar el gràfic com a PNG
plt.close()


#Gràfic d'energia amb les formules analítiques d'alçades i altres ------------------------------------------------------------------------------------

# Paràmetres inicials
latitud = np.radians(41.3888)  # Latitud de Barcelona en radians
epsilon = np.radians(23.44)  # Inclinació axial de la Terra en radians
area_panel = 2.0  # Àrea del panell solar en m²
intervals_per_dia = 24  # Hores per dia

# Llistes de posicions de la Terra vista des del Sol
# Exemple de llistes (substitueix pels teus dades reals)
x_list = np.array(x_rk4_llista)  # Posició x en unitats astronòmiques
y_list = np.array(y_rk4_llista)  # Posició y en unitats astronòmiques

# Convertir posicions a distància i angle orbital
distancies = np.sqrt(x_list**2 + y_list**2)  # Distància Terra-Sol en UA

lambda_list = np.arctan2(y_list, -x_list)
lambda_list = np.mod(lambda_list, 2 * np.pi)

# Calcular declinació solar (δ) per a cada posició
delta_list = np.arcsin(np.sin(epsilon) * np.sin(lambda_list))

# Preparar variables per emmagatzemar resultats
energia_diaria = []
energies_per_hora = []

# Calcular energia per hora durant un any complet
for dia in range(len(x_list) // intervals_per_dia):
    energia_total_dia = 0
    for hora in range(intervals_per_dia):
        index = dia * intervals_per_dia + hora
        delta = delta_list[index]  # Declinació solar per a l'hora actual

        # Angle horari (H)
        H = np.radians(15 * (hora - 12))  # Angle horari en radians

        # Altitud solar (h)
        h = np.arcsin(
            np.sin(latitud) * np.sin(delta) + np.cos(latitud) * np.cos(delta) * np.cos(H)
        )
        # Només considerar si el Sol està per sobre de l'horitzó
        if h > 0:
            cos_theta = np.sin(h)  # Correcció de l'angle d'incidència
            P = 3.828*10**26  # watts de potència solar, lluminositat solar
            irradiancia = P / (4 * np.pi * (distancies[index] * l0)**2)
            P_rebuda = area_panel * cos_theta * irradiancia * 10**(-3)  # en kW
            P_aprofitada = 4 / 10 * P_rebuda
            energia = P_aprofitada  # kWh
        else:
            energia = 0  # No hi ha energia si el Sol està sota l'horitzó

        energies_per_hora.append(energia)
        energia_total_dia += energia

    energia_diaria.append(energia_total_dia)

# Representar gràficament l'energia generada per dia
dies = np.arange(1, len(energia_diaria) + 1)
plt.show()
# Gràfic: Energia diària generada pels panells solars a Barcelona
plt.figure(figsize=(12, 6))
plt.plot(dies, energia_diaria, label="Energia diària")
plt.xlabel("Dia de l'any")
plt.ylabel("Energia generada (kWh)")
plt.title("Energia diària generada pels panells solars a Barcelona (fórmules analítiques)")
plt.grid()
plt.legend()
plt.savefig("energia_analític.png")  # Guardar el gràfic com a PNG
plt.close()


# Resultat total
energia_total_anual = sum(energia_diaria)
print(f"Energia total generada en un any (analític): {energia_total_anual:.2f} kWh")

#Gràfic energies amb dades de la llibreria astropy-------------------------------------------------------------------------------------------------------------------------------

# Paràmetres de Barcelona
latitud = 41.38879  # Latitud de Barcelona en graus (41.38879° N)
longitud = 2.15899  # Longitud de Barcelona en graus (2.15899° E)
altura = 12  # Altura sobre el nivell del mar en metres (aproximada)

# Ubicació de l'observador
ubicacio = EarthLocation(lat=latitud, lon=longitud, height=altura)

# Configuració de dates (un any complet)
data_inici = "2024-01-01 00:00:00"
data_fi = "2024-12-31 23:59:59"
intervals_per_dia = 24  # Interval per dia (cada hora)

# Generar temps per a un any complet
temps = Time(
    np.linspace(Time(data_inici).jd, Time(data_fi).jd, 365 * intervals_per_dia), format="jd"
)

# Marc de referència AltAz
altaz_frame = AltAz(location=ubicacio, obstime=temps)
sol_posicions = get_sun(temps).transform_to(altaz_frame)

# Extreure altitud del Sol
altituds = sol_posicions.alt.deg  # En graus

# Paràmetres del panell solar
area_panel = 2.0  # en m²

# Calcular l'energia generada per hora durant un any complet
energies_per_hora = []
dies = np.arange(1, 366)  # Dies de l'any
energia_per_dia = []
energia_acumulada = 0

for i, altitud in enumerate(altituds):
    if altitud > 0:  # Només si el Sol està per sobre de l'horitzó
        cos_theta = np.cos(np.radians(90 - altitud))  # Angle d'incidència
        P = 3.828 * 10**26  # watts de potència solar, lluminositat solar
        irradiancia = P / (4 * np.pi * (distancies[i] * l0)**2)
        P_rebuda = area_panel * cos_theta * irradiancia * 10**(-3)  # en kW
        P_aprofitada = 4 / 10 * P_rebuda
        energia = P_aprofitada  # kWh
    else:
        energia = 0  # Sense generació d'energia si el Sol està sota l'horitzó
    energies_per_hora.append(energia)

    # Acumular energia diària
    if (i + 1) % intervals_per_dia == 0:  # Final del dia
        energia_total_dia = sum(energies_per_hora[-intervals_per_dia:])
        energia_per_dia.append(energia_total_dia)
        energia_acumulada += energia_total_dia

# Resultats
print(f"Energia total generada el 2024 (Astropy): {energia_acumulada:.2f} kWh\n")

"""
print("Energia generada per dia (kWh):")
for dia, energia in zip(dies, energia_per_dia):
   print(f"Dia {dia}: {energia:.2f} kWh")
"""

# Representar gràficament l'energia generada durant l'any

# Gràfic: Energia generada pels panells solars a Barcelona (Astropy 2024)
plt.figure(figsize=(12, 6))
plt.plot(dies, energia_per_dia, label="Energia diària")
plt.xlabel("Dia de l'any")
plt.ylabel("Energia generada (kWh)")
plt.title("Energia generada pels panells solars a Barcelona (Astropy 2024)")
plt.grid()
plt.legend()
plt.savefig("energia__astropy_2024.png")  # Guardar el gràfic com a PNG
plt.close()

#Gràfic analític analemas ------------------------------------------------------------------------------------------------------------------------------
# Constants
LATITUD = 41.38  # Latitud de Barcelona en graus
LONGITUD = 2.17  # Longitud de Barcelona en graus
INCLINACIO = 23.44  # Inclinació de l'eix terrestre en graus
DIES_ANY = 365  # Aproximació (sense considerar anys de traspàs)
def declinacio_solar(dia_de_l_any):
    """Calcula la declinació solar en graus per a un dia donat."""
    return INCLINACIO * np.sin(np.radians((360 / DIES_ANY) * (dia_de_l_any - 81)))

def equacio_del_temps(dia_de_l_any):
    """Aproximació de l'equació del temps en minuts."""
    b = 2 * np.pi * (dia_de_l_any - 81) / 364
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

def calcular_vector_solar(dia_de_l_any, hora_utc):
    """Calcula el vector solar unitari (Sx, Sy, Sz) per a un dia i hora donats."""
    declinacio = declinacio_solar(dia_de_l_any)  # Declinació solar (delta)
    lat_observador = np.radians(LATITUD)  # Latitud de l'observador (phi_o)
    long_observador = np.radians(LONGITUD)  # Longitud de l'observador (lambda_o)

    # Latitud i longitud del punt subsolar
    phi_s = np.radians(declinacio)  # Latitud subsolar (phi_s = delta)
    E_min = equacio_del_temps(dia_de_l_any)  # Equació del temps
    lambda_s = np.radians(-15 * (hora_utc - 12 + E_min / 60))  # Longitud subsolar (lambda_s)

    # Components del vector solar
    Sx = np.cos(phi_s) * np.sin(lambda_s - long_observador)
    Sy = np.cos(phi_s) * np.cos(lambda_s - long_observador) * np.sin(lat_observador) - np.sin(phi_s) * np.cos(lat_observador)
    Sz = np.cos(phi_s) * np.cos(lambda_s - long_observador) * np.cos(lat_observador) + np.sin(phi_s) * np.sin(lat_observador)

    return Sx, Sy, Sz

# Generar dades
dies = range(1, DIES_ANY + 1)
hores_utc = range(0, 24)  # Dividir el dia en punts horaris

Sx_llista = []
Sy_llista = []
Sz_llista = []
colors = []

for dia in dies:
    for hora in hores_utc:
        Sx, Sy, Sz = calcular_vector_solar(dia, hora)
        Sx_llista.append(Sx)
        Sy_llista.append(Sy)
        Sz_llista.append(Sz)
        colors.append(dia)  # Color basat en el dia de l'any

# Crear el gràfic interactiu amb Plotly
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=Sx_llista,
    y=Sy_llista,
    z=Sz_llista,
    mode='markers',
    marker=dict(
        size=2,
        color=colors,
        colorscale='Jet',
        colorbar=dict(title='Dia de l\'Any'),
        opacity=0.8
    )
))

# Configurar etiquetes i disseny
fig.update_layout(
    scene=dict(
        xaxis_title='Sx (Est-Oest)',
        yaxis_title='Sy (Nord-Sud)',
        zaxis_title='Sz (Amunt-Avall)'
    ),
    title='Analema 3D Solar Interactiu per a Barcelona'
)
# Guardar el gràfic com a imatge PNG
#fig.write_image("analema_3D_solar_barcelona.png") #SOLS DESTAGGEJAR SI TENS INSTALAT KALEIDO
fig.show(renderer="browser") #Depén on s'executi el codi cal afegir renderer="browser" a dins el parèntesi per visualitzar correctament el gràfic

# APARTAT DE MILLORES-----------------------------------------------------------------------

# Gràfic optimitzant el número de plaques per habitatge
print((sum(energia_per_dia_rk4[:79])+sum(energia_per_dia_rk4[355:]))/89,'amb una quantitat de plaques n=', 9/(sum(energia_per_dia_rk4[266:355])/89))
print(sum(energia_per_dia_rk4[79:172])/93,'amb una quantitat de plaques n=', 9/(sum(energia_per_dia_rk4[79:172])/93))
print(sum(energia_per_dia_rk4[172:266])/94,'amb una quantitat de plaques n=', 9/(sum(energia_per_dia_rk4[172:266])/94))
print(sum(energia_per_dia_rk4[266:355])/89,'amb una quantitat de plaques n=', 9/(sum(energia_per_dia_rk4[266:355])/89))

optimització = ([x*2.043 for x in energia_per_dia_rk4[:79]] + [x*1.14 for x in energia_per_dia_rk4[79:172]] + [x*1.118 for x in energia_per_dia_rk4[172:266]] + [x*2.28 for x in energia_per_dia_rk4[266:355]] + [x*2.043 for x in energia_per_dia_rk4[355:]])

energia_per_dia_rk4_duesplaques = [x*2 for x in energia_per_dia_rk4]
energia_per_dia_rk4_optim_any = [x*1.45 for x in energia_per_dia_rk4]

plt.figure(figsize=(12, 6))
plt.plot(Dies, optimització, label="Energia diària (RK4) optimitzat per estacions")
plt.plot(Dies, energia_per_dia_rk4, label='Energia diària(RK4) per una placa')
plt.plot(Dies, energia_per_dia_rk4_duesplaques, label='Energia diària(RK4) per dues plaques')
plt.plot(Dies, energia_per_dia_rk4_optim_any, label='Energia diària(RK4) per 1.45 plaques')
plt.plot([0,365],[9,9], color='red', linestyle='--', label='Energia consumida promitjada en una casa')
plt.xlabel("Dia de l'any")
plt.ylabel("Energia generada (kWh)")
plt.title("Energia generada per diversos panells durant un any")
plt.grid()
plt.legend()
plt.savefig("energia_per_plaques.png")  # Guardar el gràfic com a PNG
plt.close()
# plt.show()

# Determinar energia anual per cada inclinació
inclinacions = np.radians(np.arange(0, 91)) #Fem una llista per incliniacions de la plca entre 0 i 90 graus
energia_anual_per_inclinacio = []

for inclinacio in inclinacions:
    energia_diaria_inc = []

    for dia in range(len(x_list) // intervals_per_dia):
        energia_total_dia_inc = 0
        for hora in range(intervals_per_dia):
            index = dia * intervals_per_dia + hora
            delta = delta_list[index]

            # Angle horari (H)
            H = np.radians(15 * (hora - 12))  # Angle horari en radians

            # Altitud solar (h)
            h = np.arcsin(
                np.sin(latitud) * np.sin(delta) + np.cos(latitud) * np.cos(delta) * np.cos(H)
            )

            if h > 0:
                P = 3.828 * 10**26  # watts de potència solar, lluminositat solar
                irradiancia = P / (4 * np.pi * (distancies[index] * l0)**2)

                # Càlcul del cos(theta) amb inclinació
                theta_inc = np.arccos(
                    np.sin(latitud) * np.sin(delta) + np.cos(latitud) * np.cos(delta) * np.cos(H)
                ) - inclinacio

                if 0 <= theta_inc <= np.pi / 2:  # Només si el Sol incideix al panell
                    cos_theta_inc = np.cos(theta_inc)
                    P_rebuda_inc = cos_theta_inc * area_panel * irradiancia * 10**(-3)
                    P_aprofitada_inc = 4 / 10 * P_rebuda_inc
                    energia_inc = P_aprofitada_inc
                else:
                    energia_inc = 0

                energia_total_dia_inc += energia_inc

        energia_diaria_inc.append(energia_total_dia_inc)

    energia_anual = sum(energia_diaria_inc)
    energia_anual_per_inclinacio.append(energia_anual)

energia_anual_per_inclinacio = np.array(energia_anual_per_inclinacio)
inclinacio_optima = inclinacions[np.argmax(energia_anual_per_inclinacio)]

# Gràfic: Energia anual en funció de la inclinació
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(inclinacions), energia_anual_per_inclinacio, marker="o")
plt.xlabel("Inclinació del panell (graus)")
plt.ylabel("Energia total anual (kWh)")
plt.title("Energia total anual en funció de la inclinació del panell solar")
plt.grid()
plt.savefig("energia_anual_en_funcio_inclinacio.png")  # Guardar el gràfic com a PNG
#plt.show()
plt.close()


print(f"La inclinació òptima és de {np.degrees(inclinacio_optima):.2f} graus.")
print(f"L'energia màxima generada (corresponent als {np.degrees(inclinacio_optima):.2f} graus d'inclinació) és de {energia_anual_per_inclinacio.max():.2f} kWh.")

#Gràfic d'errors del mètodes---------------------------------------------------------------------------------------------------------------------
#Tornem a executar Euler i RK per fer-ho ara sense el desfasament inicial

# Condicions inicials
x0 = 0.98329  # Posició inicial en x (UA)
y0 = 0.0  # Posició inicial en y (UA)
vx0 = 0.0  # Velocitat inicial en x (normalitzada)
vy0 = 30.291e3 / v0  # Velocitat inicial en y (normalitzada)

# Inicialització de variables
x_euler, y_euler, vx_euler, vy_euler = x0, y0, vx0, vy0
x_rk4, y_rk4, vx_rk4, vy_rk4 = x0, y0, vx0, vy0

# Llistes per emmagatzemar les solucions
x_euler_llista, y_euler_llista = [x_euler], [y_euler]
vx_euler_llista, vy_euler_llista = [vx_euler], [vy_euler]
x_rk4_llista, y_rk4_llista = [x_rk4], [y_rk4]
vx_rk4_llista, vy_rk4_llista = [vx_rk4], [vy_rk4]

# Funcions auxiliars
def acceleracions(x, y):
    r = np.sqrt(x**2 + y**2)
    ax = -x / r**3
    ay = -y / r**3
    return ax, ay


# Paràmetres comuns
N_passos = 8760  # Nombre de passos d'integració (hores en un any)
dt = 2 * np.pi / N_passos  # Pas de temps en anys fraccionats


for i in range(N_passos-1):
    # Euler
    ax_euler, ay_euler = acceleracions(x_euler, y_euler)
    vx_euler += ax_euler * dt
    vy_euler += ay_euler * dt
    x_euler += vx_euler * dt
    y_euler += vy_euler * dt

    x_euler_llista.append(x_euler)
    y_euler_llista.append(y_euler)
    vx_euler_llista.append(vx_euler)
    vy_euler_llista.append(vy_euler)
    # RK4
    ax1, ay1 = acceleracions(x_rk4, y_rk4)
    k1_vx, k1_vy = ax1 * dt, ay1 * dt
    k1_x, k1_y = vx_rk4 * dt, vy_rk4 * dt

    ax2, ay2 = acceleracions(x_rk4 + 0.5 * k1_x, y_rk4 + 0.5 * k1_y)
    k2_vx, k2_vy = ax2 * dt, ay2 * dt
    k2_x, k2_y = (vx_rk4 + 0.5 * k1_vx) * dt, (vy_rk4 + 0.5 * k1_vy) * dt

    ax3, ay3 = acceleracions(x_rk4 + 0.5 * k2_x, y_rk4 + 0.5 * k2_y)
    k3_vx, k3_vy = ax3 * dt, ay3 * dt
    k3_x, k3_y = (vx_rk4 + 0.5 * k2_vx) * dt, (vy_rk4 + 0.5 * k2_vy) * dt

    ax4, ay4 = acceleracions(x_rk4 + k3_x, y_rk4 + k3_y)
    k4_vx, k4_vy = ax4 * dt, ay4 * dt
    k4_x, k4_y = (vx_rk4 + k3_vx) * dt, (vy_rk4 + k3_vy) * dt

    vx_rk4 += (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
    vy_rk4 += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
    x_rk4 += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    y_rk4 += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6

    x_rk4_llista.append(x_rk4)
    y_rk4_llista.append(y_rk4)
    vx_rk4_llista.append(vx_rk4)
    vy_rk4_llista.append(vy_rk4)

a =1 / (2 / ((np.sqrt(x_euler_llista[0]**2+y_euler_llista[0]**2 ))) -((vx_euler_llista[0]**2+vy_euler_llista[0]**2 )) )  # Semieix major en UA
e=0.0167
p = (1 - e**2)  # Semilatus rectum en UA

# Equació de l'òrbita en coordenades polars
def r(theta):
    return p / (1 + e * np.cos(theta))


# Generar dades de l'òrbita
n_points = 8760  # Nombre de punts per al gràfic (major resolució)
theta = np.linspace(0, 2 * np.pi, n_points)  # Angle polar en radians
r_values = r(theta)  # Distància radial per a cada valor de theta

# Convertir a coordenades cartesianes
x = (r_values * np.cos( theta) - a * e )
y = (r_values * np.sin( theta) )

# Càlcul d'errors
error_euler = np.sqrt((np.array(x_euler_llista) - x) ** 2 + (np.array(y_euler_llista) - y) ** 2)
error_rk4 = np.sqrt((np.array(x_rk4_llista) - x) ** 2 + (np.array(y_rk4_llista) - y) ** 2)
# Centrem els errors perquè comencin a 0
error_euler -= error_euler[0]  # Restem el valor inicial
error_rk4 -= error_rk4[0]  # Restem el valor inicial
# Gràfic dels errors
plt.figure(figsize=(10, 5))
plt.plot(error_euler, label="Error Euler")
plt.plot(error_rk4, label="Error RK4")
plt.title("Errors numèrics")
plt.xlabel("Hores")
plt.ylabel("Error (UA)")
plt.legend()
plt.grid()
plt.savefig("errors.png")  # Guardar el gràfic com a fitxer PNG
plt.close()

# Factor de serralada, on a les primeres hores del dia s'aprofita un % menor de la placa, fins a una altitud = 15 p.e., on ja no hi afecta la serralada.

luminositat_solar = 3.828e26
energia_per_hora_euler_serralada, energia_per_dia_euler_serralada, energia_acumulada_euler_serralada = [], [], 0
energia_per_hora_rk4_serralada, energia_per_dia_rk4_serralada, energia_acumulada_rk4_serralada = [], [], 0


for i, altitud in enumerate(altituds_rk4):
    if altitud > 0 and not (0 < altitud < 15):
        cos_theta = np.cos(np.radians(90 - altitud))
        irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
        potencia_rebuda = area * cos_theta * irradiancia * 1e-3
        potencia_utilitzada = 0.4 * potencia_rebuda
        energia_rk4_serralada = potencia_utilitzada
    elif 0 < altitud < 15:
        cos_theta = np.cos(np.radians(90 - altitud))
        irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
        potencia_rebuda = area * cos_theta * irradiancia * 1e-3
        potencia_utilitzada = 0.4 * potencia_rebuda
        energia_rk4_serralada = altitud * potencia_utilitzada /15
    else:
        energia_rk4_serralada = 0
    energia_per_hora_rk4_serralada.append(energia_rk4_serralada)
    if (i + 1) % 24 == 0:
        energia_total_dia_rk4_serralada = sum(energia_per_hora_rk4_serralada[-24:])
        energia_per_dia_rk4_serralada.append(energia_total_dia_rk4_serralada)
        energia_acumulada_rk4_serralada += energia_total_dia_rk4_serralada

for i, altitud in enumerate(altituds_euler):
    if altitud > 0 and not (0 < altitud < 15):
        cos_theta = np.cos(np.radians(90 - altitud))
        irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
        potencia_rebuda = area * cos_theta * irradiancia * 1e-3
        potencia_utilitzada = 0.4 * potencia_rebuda
        energia_euler_serralada = potencia_utilitzada
    elif 0 < altitud < 15:
        cos_theta = np.cos(np.radians(90 - altitud))
        irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
        potencia_rebuda = area * cos_theta * irradiancia * 1e-3
        potencia_utilitzada = 0.4 * potencia_rebuda
        energia_euler_serralada = altitud * potencia_utilitzada /15
    else:
        energia_euler_serralada = 0
    energia_per_hora_euler_serralada.append(energia_euler_serralada)

    if (i + 1) % 24 == 0:
        energia_total_dia_euler_serralada = sum(energia_per_hora_euler_serralada[-24:])
        energia_per_dia_euler_serralada.append(energia_total_dia_euler_serralada)
        energia_acumulada_euler_serralada += energia_total_dia_euler_serralada

# Gràfic extra: Energia generada durant l'any (simulació)
plt.figure(figsize=(12, 6))
plt.plot(Dies, energia_per_dia_rk4, label="Energia diària (RK4)")
plt.plot(Dies, energia_per_dia_euler, label="Energia diària (Euler)")
plt.plot(Dies, energia_per_dia_rk4_serralada, label="Energia diària (RK4) amb serralada")
plt.plot(Dies, energia_per_dia_euler_serralada, label="Energia diària (Euler) amb serralada")
plt.xlabel("Dia de l'any")
plt.ylabel("Energia generada (kWh)")
plt.title("Energia generada pel panell solar durant un any (simulació)")
plt.grid()
plt.legend()
plt.savefig("energia_simulacio_serralada.png")  # Guardar el gràfic com a fitxer PNG
plt.close()



# Factor d'ombra, on s'ha considerat una ombra gradual cada certes hores del dia i que només arriba a tapar com a màxim un 20% de la placa.

altitud_aparicio_ombra = 0 #Valor on apareix una ombra
altitud_desaparicio_ombra = 50 #Valor on desapareix una ombra

valor_ombra_fixa_1 = 20
valor_ombra_fixa_2 = 40 #Aquests dos valors fan al·lusió a la presència de l'ombra sencera, tapa la mateixa quantitat de llum quan el sol té un angle entre 20 i 40.

luminositat_solar = 3.828e26
energia_per_hora_euler_ombra, energia_per_dia_euler_ombra, energia_acumulada_euler_ombra = [], [], 0
energia_per_hora_rk4_ombra, energia_per_dia_rk4_ombra, energia_acumulada_rk4_ombra = [], [], 0


for i, altitud in enumerate(altituds_rk4):
    if 2 <= (i+1) % 24 <=6:   #En el nostre cas hem considerat que a la hora zero d'un dia comença a haver-hi llum, per tant les hores associades al matí on hi apareix l'ombra són aquelles entre les 2 i les 6 p.e..
      if altitud > 0 and not (altitud_aparicio_ombra < altitud < altitud_desaparicio_ombra):
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_rk4_ombra = potencia_utilitzada
      elif (altitud_aparicio_ombra < altitud < valor_ombra_fixa_1):
          factor_divisor_1 = 1 - 0.6/(valor_ombra_fixa_1 - altitud_aparicio_ombra) * (altitud-altitud_aparicio_ombra)
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_rk4_ombra = potencia_utilitzada * factor_divisor_1
      elif (valor_ombra_fixa_1 < altitud < valor_ombra_fixa_2):
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_rk4_ombra = potencia_utilitzada * 0.4
      elif (valor_ombra_fixa_2 < altitud < altitud_desaparicio_ombra):
          factor_divisor_2 = 1 - 0.6/(altitud_desaparicio_ombra - valor_ombra_fixa_2) * (altitud_desaparicio_ombra - altitud)
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_rk4_ombra = potencia_utilitzada * factor_divisor_2
      energia_per_hora_rk4_ombra.append(energia_rk4_ombra)
    else:
      if altitud > 0:
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_rk4_llista[i % len(x_rk4_llista)]**2 + y_rk4_llista[i % len(y_rk4_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_rk4_ombra = potencia_utilitzada
      else:
          energia_rk4_ombra = 0
      energia_per_hora_rk4_ombra.append(energia_rk4_ombra)
    if (i + 1) % 24 == 0:
        energia_total_dia_rk4_ombra = sum(energia_per_hora_rk4_ombra[-24:])
        energia_per_dia_rk4_ombra.append(energia_total_dia_rk4_ombra)
        energia_acumulada_rk4_ombra += energia_total_dia_rk4_ombra


for i, altitud in enumerate(altituds_euler):
    if 2 <= (i+1) % 24 <=6:
      if altitud > 0 and not (altitud_aparicio_ombra < altitud < altitud_desaparicio_ombra):
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_euler_ombra = potencia_utilitzada
      elif (altitud_aparicio_ombra < altitud < valor_ombra_fixa_1):
          factor_divisor_1 = 1 - 0.6/(valor_ombra_fixa_1 - altitud_aparicio_ombra) * (altitud-altitud_aparicio_ombra)
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_euler_ombra = potencia_utilitzada * factor_divisor_1
      elif (valor_ombra_fixa_1 < altitud < valor_ombra_fixa_2):
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_euler_ombra = potencia_utilitzada * 0.4
      elif (valor_ombra_fixa_2 < altitud < altitud_desaparicio_ombra):
          factor_divisor_2 = 1 - 0.6/(altitud_desaparicio_ombra - valor_ombra_fixa_2) * (altitud_desaparicio_ombra - altitud)
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_euler_ombra = potencia_utilitzada * factor_divisor_2
      energia_per_hora_euler_ombra.append(energia_euler_ombra)
    else:
      if altitud > 0:
          cos_theta = np.cos(np.radians(90 - altitud))
          irradiancia = luminositat_solar / (4 * np.pi * (np.sqrt(x_euler_llista[i % len(x_euler_llista)]**2 + y_euler_llista[i % len(y_euler_llista)]**2) * UA)**2)
          potencia_rebuda = area * cos_theta * irradiancia * 1e-3
          potencia_utilitzada = 0.4 * potencia_rebuda
          energia_euler_ombra = potencia_utilitzada
      else:
          energia_euler_ombra = 0
      energia_per_hora_euler_ombra.append(energia_euler_ombra)
    if (i + 1) % 24 == 0:
        energia_total_dia_euler_ombra = sum(energia_per_hora_euler_ombra[-24:])
        energia_per_dia_euler_ombra.append(energia_total_dia_euler_ombra)
        energia_acumulada_euler_ombra += energia_total_dia_euler_ombra

# Gràfic extra: Energia generada durant l'any (simulació) amb ombra:
plt.figure(figsize=(12, 6))
plt.plot(range(0,400), energia_per_hora_rk4_ombra[:400])
plt.plot(range(0,400), energia_per_hora_euler_ombra[:400])
plt.xlabel("Hores de l'any")
plt.ylabel("Energia generada (kWh)")
plt.title("Energia generada pel panell solar durant les primeres 400h d'any amb presència d'una ombra (simulació)")
plt.grid()
plt.legend()
plt.savefig("energia_simulacio_ombra.png")  # Guardar el gràfic com a fitxer PNG
plt.close()
