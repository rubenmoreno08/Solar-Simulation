# Solar & Orbital Dynamics Simulation

![Language](https://img.shields.io/badge/Language-Python%203.x-blue)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Astropy%20%7C%20Plotly-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## Descripció i Objectiu

Aquest projecte simula i visualitza la dinàmica orbital de la Terra i la radiació solar incident en un panell situat a **Barcelona**[cite: 1].

El codi combina mètodes numèrics clàssics amb dades astrofísiques d'alta precisió per estudiar l'eficiència energètica fotovoltaica sota diferents condicions, incloent-hi l'impacte d'ombres orogràfiques[cite: 1].

### Informe Científic

Per veure la base teòrica detallada, el desenvolupament matemàtic i la discussió profunda dels resultats, consulteu el document adjunt:
**[Llegir l'Informe Complet (PDF)](./Informe.pdf)**[cite: 1]

---

## Desenvolupament Tècnic

### Mètodes Numèrics
* **Mètode d'Euler**: Implementat per a la integració inicial i l'anàlisi de la propagació de l'error[cite: 1].
* **Runge-Kutta 4 (RK4)**: Utilitzat per garantir la conservació de l'energia orbital i una precisió superior en la trajectòria[cite: 1].

### Optimització Energètica
* **Càlcul de la inclinació òptima**: S'ha determinat que 27.00 graus és l'angle ideal per maximitzar la captació anual a Barcelona[cite: 1].
* **Simulació d'ombres**: Modelatge de pèrdues per obstacles físics i el relleu de la zona[cite: 1].

---

## Fitxers de Sortida

| Categoria | Descripció |
| :--- | :--- |
| **Dinàmica** | Comparativa d'òrbites, conservació de l'energia i anàlisi d'errors[cite: 1]. |
| **Energia** | Gràfics de generació diària, comparativa de plaques i estudi de *tilt*[cite: 1]. |
| **Visualització** | Analema solar de Barcelona i simulació 3D interactiva[cite: 1]. |

---

## Autors

Projecte desenvolupat per l'equip de Física (UAB):
* **Rubén Moreno**[cite: 1]
* **Joel Sánchez**[cite: 1]
* **Xavier Montero**[cite: 1]
* **Arnau Rodríguez**[cite: 1]

---

## Requeriments
```bash
pip install numpy matplotlib astropy plotly
