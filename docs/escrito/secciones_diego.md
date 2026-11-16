# Secciones de Diego: 4.4 (modelo C), 4.6 y 4.7

Borrador en Markdown para pasar después al .docx con `formato-tarea-docx`.

---

## 4.4 Metodología de desarrollo (modelo C)

### Arquitectura

El modelo C combina XLS-R 300M con una cabeza AASIST, siguiendo a Tak et al. (2022). XLS-R es
un modelo de voz autosupervisado de la familia wav2vec 2.0, preentrenado por Meta con unas
436,000 horas de audio en 128 idiomas sin etiquetas. No es un modelo de lenguaje: aprende a
representar la señal de voz prediciendo partes enmascaradas del propio audio. Por cada 20 ms de
audio entrega un vector de 1024 dimensiones, que una capa lineal reduce a 128.

Sobre esa secuencia trabaja AASIST, una red de grafos con atención que modela por separado las
relaciones entre bandas espectrales y entre instantes de tiempo, y después las combina. Su
salida son dos logits (genuina y sintética). La elección se apoya en tres temas vistos en clase:
transfer learning con un modelo preentrenado, mecanismos de atención y redes convolucionales
(los bloques residuales de la cabeza). El modelo completo tiene 315.9 M de parámetros, de los
cuales 0.45 M pertenecen a la cabeza.

La entrada es audio mono a 16 kHz recortado a 64,600 muestras (unos 4 s), sin normalizar la
amplitud en el dataset; la normalización a media cero y varianza uno por audio se hace dentro
del modelo, igual que en el preentrenamiento de XLS-R.

### Iteraciones

Se entrenaron seis variantes para medir por separado el efecto del fine-tuning, de la
augmentation y de los datos en español.

| Iteración | Qué cambia respecto a la anterior | Parámetros entrenados |
|---|---|---|
| C0 | XLS-R congelado; solo se entrena la cabeza AASIST (modelo base del transfer learning) | 0.45 M |
| C1 | Fine-tuning completo de XLS-R y la cabeza desde cero | 315.9 M |
| C1b | LP-FT: parte de C0 y ajusta todo con tasas de aprendizaje bajas | 315.9 M |
| C2 | C0 con augmentation telefónica y RawBoost (probabilidad 0.5 cada una) | 0.45 M |
| C3 | C2 reentrenado con voz real en español (FLEURS) y sintetizadores modernos (Edge TTS y F5-TTS) | 0.45 M |
| C4 | C3 reentrenado 15 épocas con 1,500 clones OmniVoice (2025) de hablantes de FLEURS | 0.45 M |

C0 existe para responder la pregunta que plantea el enunciado: cuánto aporta ajustar el modelo
preentrenado frente a usarlo como extractor fijo de características.

### Función de pérdida

Todas las variantes usan entropía cruzada ponderada. ASVspoof 2019 LA tiene un 10 % de audios
genuinos en train, así que cada clase recibe un peso inverso a su frecuencia (2.0 para genuina
y 0.56 para sintética en ASVspoof; los pesos se recalculan en C3 y C4 con el conjunto combinado).
Sin esa ponderación el modelo minimiza la pérdida declarando casi todo sintético, lo que en el
banco se traduce en transferir a clientes legítimos.

La pérdida no fija el punto de operación. El modelo entrega un score y la decisión se toma
después con dos umbrales elegidos por el modelo de valor de la sección 4.3, que pondera el
fraude evitado contra el costo de cada alerta y el costo de cómputo. Así, un cambio en los costos
del banco mueve los umbrales sin reentrenar.

### Optimización e hiperparámetros

Todas las corridas usan Adam con weight decay de 1e-4, scheduler coseno, precisión bf16, batch
efectivo de 16 y semilla 42. La parada temprana mide el EER en dev al final de cada época y
detiene el entrenamiento si no mejora en `patience` épocas, guardando el mejor checkpoint.

| Iteración | lr XLS-R | lr cabeza | Épocas máx. / paciencia | Batch (× acumulación) |
|---|---|---|---|---|
| C0 | congelado | 1e-4 | 10 / 3 | 16 |
| C1 | 1e-6 | 1e-4 | 10 / 3 | 4 × 4, gradient checkpointing |
| C1b | 1e-6 | 1e-5 | 10 / 3 | 4 × 4, gradient checkpointing |
| C2 | congelado | 1e-4 | 10 / 3 | 16 |
| C3 | congelado | 5e-5 | 6 / 2 | 16 |
| C4 | congelado | 5e-5 | 15 / 4 | 16 |

Las tasas de C1 (1e-6 para XLS-R y 1e-4 para la cabeza) son las de Tak et al. (2022). No se hizo
búsqueda de hiperparámetros por grilla: con 3 horas por corrida de fine-tuning en una sola GPU
de 8 GB, se priorizó comparar estrategias de transfer learning (congelar, ajustar todo, LP-FT)
antes que afinar tasas dentro de una misma estrategia.

La augmentation de C2 tiene dos partes. La telefónica remuestrea a 8 kHz y vuelve a 16 kHz, y
aplica un filtro pasa banda de 300 a 3,400 Hz, que es lo que deja pasar una línea telefónica.
RawBoost agrega ruido coloreado y distorsiones convolutivas y no lineales sobre la forma de
onda. Cada una se aplica a la mitad de los audios para que el modelo siga viendo voz de banda
ancha.

### Costo computacional

Todo el modelo C se entrenó en una laptop con RTX 5070 Laptop (8 GB). El costo se reporta con el
precio de una GPU de nube equivalente (clase L4, 0.70 USD/h).

| Iteración | Horas | Costo (USD) | VRAM | EER dev | EER eval (IC 95 %) | min t-DCF eval |
|---|---|---|---|---|---|---|
| C0 | 0.55 | 0.39 | 3.1 GB | 0.17 % | 1.08 % (0.96 a 1.23) | 0.0347 |
| C1 | 3.13 | 2.19 | 5.9 GB | 0.85 % | 3.27 % (3.05 a 3.47) | 0.0973 |
| C1b | 1.96 | 1.37 | 5.9 GB | 0.57 % | 1.61 % (1.47 a 1.79) | 0.0527 |
| C2 | 1.35 | 0.95 | 3.1 GB | 0.20 % | 0.99 % (0.88 a 1.12) | 0.0269 |
| C3 | 0.91 | 0.63 | 3.2 GB | 0.42 %* | 1.37 % (1.26 a 1.49) | 0.0396 |
| C4 | 2.78 | 1.94 | 3.2 GB | 0.66 %* | 2.16 % (2.01 a 2.32) | 0.0607 |

\* El dev de C3 y C4 combina ASVspoof 2019 con el dev en español, así que no se compara
directamente con el de C0 a C2.

### Lo que mostró la comparación del fine-tuning

El fine-tuning completo empeoró el modelo. C1 bajó el EER en dev pero triplicó el de eval (de
1.08 % en C0 a 3.27 %), y en el ataque más difícil, A10, pasó de 3.12 % a 10.11 %. El conjunto
de train de ASVspoof solo tiene 6 sintetizadores, y con 316 M de parámetros libres el modelo se
especializa en ellos y distorsiona las características generales que XLS-R aprendió en el
preentrenamiento. LP-FT (C1b), que primero entrena la cabeza y después ajusta todo con tasas
bajas, redujo el daño (1.61 %) pero no superó a C0. Por eso las iteraciones siguientes mantienen
XLS-R congelado: es más barato (0.45 M de parámetros, 3.1 GB de VRAM), más rápido de entrenar y
generaliza mejor a ataques que no vio.

La mejora que sí funcionó fue la augmentation. C2 iguala a C0 en dev pero gana en eval (EER
0.99 %, min t-DCF 0.0269, 23 % menos que C0) y, sobre todo, en canal telefónico: con códecs
G.711, GSM y AMR el EER de C0 sube a 4.47 % y el de C2 queda en 2.12 %, con intervalos que no se
traslapan.

C3 se agregó después de probar el sistema con voces en español (sección 4.7): C2 detectaba
sintetizadores clásicos, pero solo el 70 % de los audios de Edge TTS y ningún clon de un servicio
comercial. Reentrenar la cabeza con FLEURS (voz real en español de Latinoamérica), Edge TTS y
F5-TTS bajó el EER en el conjunto de prueba en español de 14.6 % a 3.7 %, con voces y hablantes
que no estaban en train (medido antes de agregar OmniVoice a ese conjunto). El costo fue un olvido parcial de ASVspoof (EER eval de 0.99 % a
1.37 %), el comportamiento esperado al ajustar un modelo a una distribución nueva.
C4 agregó al entrenamiento 1,500 clones generados con OmniVoice, un clonador de 2025, a partir
de hablantes de FLEURS. Con OmniVoice incluido en el conjunto de prueba en español, el EER queda
así: C2 24.2 %, C3 18.4 % y C4 4.0 % (IC 95 % de 3.2 a 5.1). Por generador, C4 baja OmniVoice de
30.8 % a 7.9 % y F5-TTS de 5.5 % a 1.1 %, y mantiene Edge TTS en 0.2 %. El precio es un olvido
mayor de ASVspoof: el EER en eval sube a 2.16 % y en canal telefónico a 3.25 %, frente a 0.99 % y
2.12 % de C2. C4 detecta 5 de 11 clones OmniVoice de la voz del integrante, que C2 y C3 dejaban
pasar todos, pero sigue sin detectar ninguno del servicio comercial (sección 4.7).

---

## 4.6 Despliegue y experiencia de usuario

### Arquitectura del sistema

El producto, llamado Signet, es una aplicación web que usa el agente del centro de contacto
durante la llamada. Todo corre en un contenedor Docker desplegado en Hugging Face Spaces.

```
Navegador del agente (React)
  graba el micrófono (sin cancelación de eco ni control de ganancia) o recibe un archivo
        │  POST /analyze (audio + prevalencia)
        ▼
API FastAPI
  1. decodifica con ffmpeg a mono 16 kHz (WAV, FLAC, WebM, OGG, MP3), hasta 20 s
  2. revisa calidad: audio corto, silencio, saturación
  3. corta ventanas de 4 s y marca cuáles tienen voz
  4. ONNX Runtime en CPU (2 hilos): score de cada ventana
  5. promedia en log-odds las ventanas con voz
  6. calibra (Platt) y ajusta a la prevalencia supuesta → P(sintética)
  7. compara el score con t1/t2 del modelo de valor → acción
        │  JSON: acción, P(sintética), scores por ventana, avisos, latencia
        ▼
Panel: sello con la acción, probabilidad, escala de umbrales, registro de la sesión
  GET /metrics → monitoreo (tasa de alertas, latencias, histograma, alarma de deriva)
```

La API no usa PyTorch: el modelo se exporta a ONNX (diferencia máxima con PyTorch de 2.5e-5) y
corre con ONNX Runtime, lo que deja una imagen de 3.1 GB que funciona en la CPU básica gratuita
de Hugging Face Spaces. El audio no se guarda en ningún momento.

El análisis por ventanas reemplazó una primera versión que solo leía los primeros 4 s. Con 8 s o
más de voz, la decisión promedia varias ventanas y un fragmento atípico (un golpe en el
micrófono, una pausa) ya no decide solo. Una ventana vota si tiene al menos 0.8 s por encima de
unos -46 dBFS. El criterio es absoluto a propósito: una revisión de código detectó que un umbral
relativo a la ventana más fuerte permitía que un ruido fuerte al inicio dejara fuera a un clon
que habla más bajo.

### Métricas operativas

| Métrica | Valor |
|---|---|
| Latencia por ventana de 4 s (ONNX, CPU 2 hilos) | p50 588 ms, p95 608 ms |
| Latencia de un audio de 8 s (2 a 3 ventanas) | aprox. 1.9 s |
| Throughput en GPU (RTX 5070, batch 32) | 54 audios/s |
| Costo de cómputo | 0.014 USD por 1,000 llamadas (CPU a 0.085 USD/h) |
| Costo mensual para 500,000 llamadas | aprox. 7 USD |

El costo de cómputo es despreciable frente al valor del sistema (unos 2.1 millones de USD al
mes con los supuestos de la sección 4.3), así que la elección del modelo depende de cuántos
fraudes detecta y cuántas falsas alarmas genera, no de su latencia.

### Cómo el modelo cambia la experiencia del agente

El panel pone primero la acción: un sello que dice Continuar, Verificar o Transferir, con
texto e ícono además del color, para que el agente decida sin leer números. Debajo aparecen la
probabilidad calibrada, dónde cayó el score respecto a los umbrales y el score de cada ventana,
para quien necesite justificar la decisión.

La diferencia entre modelos se nota en las falsas alarmas, que son las que el agente sufre. Con
audio telefónico y los umbrales elegidos en dev, C0 alerta en el 4.70 % de las llamadas
legítimas y C2 en el 0.20 %. Para 500,000 llamadas al mes, eso pasa de unas 23,500
transferencias innecesarias al analista a unas 1,000, y el costo de alertas baja de 293,000 a
23,000 USD al mes. En la prueba con grabaciones propias, C0 mandó al analista la frase real
"necesito autorizar un pago de cinco mil quetzales, es urgente"; C2 y C3 la dejaron continuar.

No se hizo una prueba formal de usabilidad con agentes reales. La evidencia disponible es la
prueba propia descrita en la sección 4.7 y la revisión del diseño del panel.

### Manejo de errores e incertidumbre

- Un archivo vacío, dañado o en un formato no soportado devuelve un error claro y no un
  veredicto.
- Con menos de 1 s de audio, en silencio o saturado, el veredicto se acompaña de un aviso
  visible de que la predicción es poco confiable.
- Si un modelo no tiene umbrales elegidos por el modelo de valor, la API los marca como
  provisionales y el panel lo muestra.
- La probabilidad se calibra para la prevalencia real de ataques (0.2 % por defecto, ajustable
  por el banco). Así un score medio sospechoso se presenta como lo que es: con 1 ataque cada
  500 llamadas, la mayoría de esos casos siguen siendo clientes reales.
- El pie del panel recuerda que el sistema es un apoyo y que la decisión final sigue los
  procedimientos del banco.

### Plan de monitoreo

`GET /metrics` lleva por modelo el número de llamadas, las acciones tomadas, la latencia p50 y
p95, el histograma de scores y la tasa reciente de alertas. Esa tasa se compara con la esperada
según el modelo de valor (0.28 % para C2), y si se aleja se enciende una alarma de deriva en el
panel. La alarma se probó en desarrollo: tras analizar varios audios sintéticos seguidos, la
tasa reciente llegó a 17 % frente al 0.28 % esperado.

La alarma detecta cambios en la mezcla de llamadas, pero no detecta un sintetizador nuevo que
el modelo confunda con voz real, porque ese caso no genera alertas. Los resultados de la
sección 4.7 muestran que es el riesgo principal, así que el plan de monitoreo incluye una
evaluación periódica con clones generados por los sintetizadores más recientes, igual a la que
se hizo en este proyecto, y reentrenar la cabeza cuando la detección baje.

---

## 4.7 Explicabilidad, ética y riesgos

### Explicabilidad

Para el modelo C se usó oclusión, un método que no depende de la arquitectura: se elimina una
parte del audio y se mide cuánto cambia la evidencia de "sintética" (el log-odds del score).
Se aplicó en frecuencia, anulando una banda del espectro y promediando sobre 40 audios de eval
por ataque, y en tiempo, silenciando ventanas de 0.25 s. Para el modelo A se usó Grad-CAM sobre
los coeficientes LFCC (sección de Cisco).

La oclusión en frecuencia explica por qué C2 aguanta mejor el canal telefónico que C0:

- C2 encuentra evidencia de voz sintética en varias bandas: 0 a 300 Hz, 2,000 a 3,400 Hz y
  3,400 a 5,000 Hz. Una de ellas está dentro de la banda telefónica, así que la evidencia
  sobrevive al códec.
- C0 casi no busca artefactos de síntesis. Decide sobre todo según si la banda de 300 a 1,000 Hz
  suena natural: al quitarla de un audio genuino, la evidencia de "sintética" sube 8.3 puntos de
  log-odds en C0 y 3.6 en C2. Cualquier distorsión de esa banda, como la de un códec, hace que C0
  dispare falsas alarmas, que es justo lo que se midió en canal telefónico.

La oclusión temporal reveló un sesgo del dataset. En un audio del ataque A10, lo único que
empujaba hacia "sintética" eran los primeros 0.25 s de silencio. En ASVspoof 2019 la duración
del silencio inicial está correlacionada con la etiqueta (Müller et al., 2021), y el modelo lo
aprovecha. Como mitigación, en el producto las ventanas sin voz no votan en la decisión.

**[Figuras: `results/figures/C2_oclusion_frecuencia.png`, `C0_oclusion_frecuencia.png` y
`C2_oclusion_tiempo_LA_E_8339197.png`.]**

### Sesgos

El sesgo más grande no es de un grupo de personas sino de época: el modelo detecta los
sintetizadores que conoce y falla con los nuevos. Para medirlo se armó un conjunto de prueba
propio con voz real de un integrante del equipo (10 frases de banca, grabadas con consentimiento)
y con clones de varias generaciones de sintetizadores.

| Audio de prueba | C2 | C3 | C4 |
|---|---|---|---|
| Voz real (micrófono) | 10/10 | 10/10 | 10/10 |
| Voz real (canal telefónico) | 10/10 | 10/10 | 10/10 |
| XTTS v2 (2023), 6 voces genéricas | 56/60 | 58/60 | 59/60 |
| XTTS v2, clon de la voz del integrante | 9/10 | 7/10 | 10/10 |
| F5-TTS (2024), clon del integrante | 9/10 | 10/10 | 10/10 |
| Edge TTS, 8 voces de Centroamérica y México no vistas | 56/80 | 79/80 | 80/80 |
| OmniVoice (2025, VoiceStudio), clon del integrante | 0/11 | 0/11 | 5/11 |
| Servicio comercial de clonación, clon del integrante | 0/5 | 0/5 | 0/5 |

C3 aprendió los sintetizadores con los que se entrenó (Edge TTS y F5-TTS) aunque las voces y los
hablantes de prueba fueran nuevos, pero no detecta OmniVoice ni el servicio comercial. C4, que
sí vio clones OmniVoice de otros hablantes, detecta 5 de los 11 de la voz del integrante, y sigue
sin detectar el servicio comercial. Un detector entrenado con ataques de una época queda ciego
ante la siguiente generación, y cerrar esa brecha exige conseguir audio del generador nuevo y
reentrenar. Es el problema abierto del área.

Hay otros sesgos medidos:

- ASVspoof 2019 es inglés, y en el conjunto de prueba en español (FLEURS) C2 tiene un EER de
  24.2 % (contando los clones OmniVoice). C3, entrenado también con español, lo baja a 18.4 %, y
  C4, que además vio OmniVoice, a 4.0 %.
- Con audio telefónico el EER de C0 se multiplica por cinco (de 0.94 % a 4.47 % sobre el mismo
  subconjunto) y el de C2 por 2.4 (de 0.89 % a 2.12 %).
- La prueba propia tiene un solo hablante, hombre y guatemalteco. No se evaluó el desempeño por
  género, edad ni acento, y los resultados no deben generalizarse a otros grupos.

El costo de estos errores no es simétrico. Una falsa alarma transfiere a un cliente legítimo al
analista de fraude, lo que significa fricción y desconfianza, y si el modelo falla más con cierto
acento o con voces mayores, esa fricción recae siempre sobre las mismas personas. Una detección
fallida deja pasar el fraude completo, con una pérdida promedio estimada de 2,500 USD.

### Riesgos de uso indebido y privacidad

- Un atacante puede probar sus clones contra el detector hasta encontrar uno que pase. Los
  resultados con OmniVoice muestran que hoy no tiene que esforzarse mucho.
- Si el banco trata "Continuar" como prueba de identidad, el sistema se vuelve un punto único de
  falla. Debe ser un factor más, junto con la verificación habitual.
- La voz es un dato biométrico. La API no guarda audio y el panel solo conserva el resultado en
  la pestaña del navegador. Las grabaciones del equipo se hicieron con consentimiento y no están
  en el repositorio.
- ASVspoof 2019 (ODC-By) y FLEURS (CC-BY 4.0) permiten su uso con atribución. XTTS v2 (CPML) y
  OmniVoice (CC-BY-NC) solo permiten uso no comercial, así que los clones generados con ellos
  sirven para investigación pero no para entrenar una versión comercial del producto.

### Cuándo no debe usarse

- Como único factor para autorizar operaciones o para bloquear a un cliente.
- Con clonadores recientes o comerciales que no estén representados en el entrenamiento: en las
  pruebas, OmniVoice y el servicio comercial pasaron sin ser detectados.
- En idiomas distintos del español y el inglés, que no se evaluaron.
- Con ataques de reproducción física (un clon reproducido por un parlante frente al micrófono),
  que pertenecen al escenario "physical access" de ASVspoof y no se entrenaron.
- Con menos de 1 s de voz, en silencio o con el audio saturado.
