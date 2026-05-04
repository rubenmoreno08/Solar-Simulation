# Simulació de Dinàmica Orbital i Radiació Solar

![Status](https://img.shields.io/badge/Project-Research-blueviolet)
![Language](https://img.shields.io/badge/Language-Python%203.x-blue)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Astropy%20%7C%20Plotly-green)
![Field](https://img.shields.io/badge/Field-Computational_Physics-red)

Aquest repositori conté una eina computacional dissenyada per avaluar la viabilitat d'instal·lacions solars mitjançant la integració de models físics i mètodes numèrics avançats.

---

## Eixos del Projecte

### 1. Motor de Simulació Orbital
Per determinar la posició del Sol amb precisió, hem modelat el sistema Terra-Sol resolent les equacions de Newton. 
* **RK4 vs Euler**: S'ha implementat l'algorisme de Runge-Kutta de 4t ordre per garantir la conservació de l'energia del sistema, comparant-ne l'estabilitat respecte al mètode d'Euler.
* **Adimensionalització**: El codi treballa amb unitats normalitzades per optimitzar el cost computacional i evitar errors d'escala.

### 2. Geometria de Coordenades Local
El pas de l'òrbita el·líptica a la placa solar requereix una cadena de transformacions lineals:
* **Equatorial a ECEF**: Ajust per la inclinació axial de la Terra.
* **ECEF a ENU**: Projecció de la posició solar sobre l'horitzó local de Barcelona utilitzant latitud i longitud.

### 3. Model Energètic i Optimització
Més enllà de la física teòrica, el projecte busca resultats pràctics:
* **Llei de Stefan-Boltzmann**: Per calcular la irradiància incident real.
* **Anàlisi de Tilt**: S'ha programat una iteració per trobar l'angle d'inclinació òptim (27 graus) que maximitza el rendiment anual.
* **Relleu i Ombres**: El model preveu l'impacte d'obstacles orogràfics en la captació d'energia.

---

## Estructura de Resultats

| Mòdul | Output Clau |
| :--- | :--- |
| **Física Numèrica** | Anàlisi d'error acumulat i conservació de l'energia orbital. |
| **Enginyeria Solar** | Corbes de producció diària i optimització del nombre de panells. |
| **Astrometria** | Generació de l'analema solar des de la perspectiva de Barcelona. |

---

## Documentació Completa

Totes les demostracions matemàtiques, l'anàlisi de dades i les conclusions físiques es troben detallades a l'informe tècnic:
**[Consulteu l'Informe.pdf](./Informe.pdf)**

---

## Equip de Desenvolupament
* **Rubén Moreno**
* **Joel Sánchez**
* **Xavier Montero**
* **Arnau Rodríguez**

---

## Setup
```bash
pip install numpy matplotlib astropy plotly
