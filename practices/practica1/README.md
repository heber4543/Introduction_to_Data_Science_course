**Descripción del dataset:**
1. Descripción: Datos del artículo *Working memory contributions to reinforcement learning impairments in schizophrenia" (Collins et al., 2014) in JoN (Journal Of Nueroscience)*
2. Acceso: OSF (Open Science Framework)
3. Date Created: Feb 22, 2023, 12:02 PM
4. Date Updated: May 8, 2023, 1:36 PM
5. Contribuidores: CCNLab UC Berkeley (https://ccn.studentorg.berkeley.edu/)
6. Enlace al artículo: https://pubmed.ncbi.nlm.nih.gov/25297101/
7. Cita: Berkeley, C. U. (2023, May 8). Data for Collins et al., 2014 JoN. Retrieved from osf.io/q67tb
8. Enlace para acceso: https://osf.io/q67tb/overview
9.  El repositorio de origen no declara una licencia explícita. Pero se cita a la fuente.
10. Breve resumen del artículo, generado por IA: El paper investiga por qué las personas con esquizofrenia muestran déficits en tareas de aprendizaje por refuerzo, proponiendo que estos no se deben principalmente a fallas en el sistema de reinforcement learning (RL), sino a alteraciones en la working memory (WM). Para ello, los autores utilizan una tarea conductual que manipula la carga de memoria (variando el número de estímulos por bloque) junto con un modelo computacional híbrido (RLWM) que separa ambos procesos. Los resultados muestran que, aunque los pacientes tienen peor desempeño global, sus parámetros de aprendizaje por refuerzo permanecen relativamente intactos; en cambio, presentan menor capacidad de memoria de trabajo, mayor olvido y menor uso de este sistema. Esto sugiere que los déficits observados en aprendizaje no son puramente dopaminérgicos o de RL, sino que emergen principalmente de limitaciones en memoria de trabajo que afectan el desempeño en estas tareas.

**Datos:**

| Columna | Explicación |
|---|---|
| subno | Identificador |
| block | Número de bloque |
| ns | Tamaño de conjunto |
| time | Número de eensayo |
| stimseq | ID del estímulo |
| imageseq | Número de imagen actual |
| folderseq | Número de la categoría de la imagen |
| iterseq | Número de iteración por estímulo |
| corAseq | Acción correcta |
| choice | Acción actual |
| key | Tecla presionada |
| cor | Es correcto? Sí/No -> 0/1 |
| rew | Recibió recompensa? Sí/No -> 0/1 |
| rt | Tiempo de reacción |
| condition (HC=0,SZ=1) | Condición experimental. 0: grupo control / 1: esquizofrenia. **Variable objetivo** |
| pcor | número de ensayos correctos previos para un estímulo dado (utilizado para regresión logística) |
| delay | demora desde el último ensayo correcto para el mismo estímulo |

**Requisitos de ejecución:**
1. Cuenta en Google Colab
2. Python 3
3. Librerías: pandas, numpy y matplotlib
4. Acceso al archivo de datos utilizado por el notebook

**Instrucciones de uso:**
La carpeta 'practica1' contiene el notebook y una carpeta con el dataset. En la carpeta 'datos' encontrarán el .csv del dataset y un readme que los autores del dataset proporcionan al descargar el dataset.
