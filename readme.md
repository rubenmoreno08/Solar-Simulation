# Anàlisi Numèrica: Optimització Fotovoltaica i Dinàmica Celeste

![Status](https://img.shields.io/badge/Project-Research-blueviolet)
![Field](https://img.shields.io/badge/Field-Computational_Physics-red)
![Location](https://img.shields.io/badge/Focus-Barcelona-yellow)

Aquest repositori conté una eina computacional dissenyada per avaluar la viabilitat d'instal·lacions solars a Barcelona mitjançant la integració de models físics i mètodes numèrics avançats[cite: 1].

---

## Eixos del Projecte

### 1. Motor de Simulació Orbital
Per determinar la posició del Sol amb precisió, hem modelat el sistema Terra-Sol resolent les equacions de Newton[cite: 1]. 
* **RK4 vs Euler**: S'ha implementat l'algorisme de Runge-Kutta de 4t ordre per garantir la conservació de l'energia del sistema, comparant-ne l'estabilitat respecte al mètode d'Euler[cite: 1].
* **Adimensionalització**: El codi treballa amb unitats normalitzades per optimitzar el cost computacional i evitar errors d'escala[cite: 1].

### 2. Geometria de Coordenades Local
El pas de l'òrbita el·líptica a la placa solar requereix una cadena de transformacions lineals[cite: 1]:
* **Equatorial a ECEF**: Ajust per la inclinació axial de la Terra[cite: 1].
* **ECEF a ENU**: Projecció de la posició solar sobre l'horitzó local de Barcelona utilitzant latitud i longitud[cite: 1].

### 3. Model Energètic i Optimització
Més enllà de la física teòrica, el projecte busca resultats pràctics[cite: 1]:
* **Llei de Stefan-Boltzmann**: Per calcular la irradiància incident real[cite: 1].
* **Anàlisi de Tilt**: S'ha programat una iteració per trobar l'angle d'inclinació òptim (27°) que maximitza el rendiment anual[cite: 1].
* **Relleu i Sombras**: El model preveu l'impacte d'obstacles orogràfics en la captació d'energia[cite: 1].

---

## Estructura de Resultats

| Mòdul | Output Clau |
| :--- | :--- |
| **Física Numèrica** | Anàlisi d'error acumulat i conservació de l'energia orbital[cite: 1]. |
| **Enginyeria Solar** | Corbes de producció diària i optimització del nombre de panells[cite: 1]. |
| **Astrometria** | Generació de l'analema solar des de la perspectiva de Barcelona[cite: 1]. |

---

## Documentació Completa

Totes les demostracions matemàtiques, l'anàlisi de dades i les conclusions físiques es troben detallades a l'informe tècnic:
📄 **[Consulteu l'Informe.pdf](./Informe.pdf)**[cite: 1]

---

## Equip de Desenvolupament
* **Rubén Moreno**[cite: 1]
* **Joel Sánchez**[cite: 1]
* **Xavier Montero**[cite: 1]
* **Arnau Rodríguez**[cite: 1]

---

## Setup
```bash
pip install numpy matplotlib astropy plotly
