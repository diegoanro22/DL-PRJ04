# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite (TypeScript) en `app/web/`. El build estático lo sirve la API FastAPI en `/`
(mismo origen que `/analyze`), y todo se despliega como un contenedor Docker en Hugging Face
Spaces. Sin backend aparte para el panel.

## Users

- **Agente del centro de contacto bancario** (usuario principal). Atiende una llamada entrante
  en la que el cliente pide una operación sensible (transferencia, cambio de datos, reposición
  de tarjeta). Necesita saber en segundos si la voz parece sintética y qué hacer: continuar,
  pedir un factor adicional o transferir al analista de fraude. No es experto en ML.
- **Jurado del curso CC3092 Deep Learning (UVG)**, en la presentación del 16 a 17 nov 2026.
  Ve la misma pantalla en vivo, con audio grabado ahí mismo (incluida la voz del catedrático) y
  con clones generados por el equipo. Debe entender el mecanismo, la incertidumbre y el costo.

## Product Purpose

Detectar voz clonada o sintética en los primeros 4 segundos de una llamada y convertir el
score en una acción operativa con umbrales elegidos por valor económico (fraude evitado contra
fricción al cliente legítimo y costo de cómputo). Éxito: el agente decide sin leer números, y
el jurado ve un flujo desplegado de punta a punta, no un notebook.

## Positioning

Modelos entrenados por el equipo (LFCC+LCNN, AASIST-L, XLS-R + AASIST), no un LLM. La salida
no es "real/falso" sino una de tres acciones con umbrales justificados por un modelo de valor
en USD, con probabilidad calibrada para la prevalencia real de ataques (≈0.5 %).

## Operating Context

- Entrada: grabación desde el micrófono del navegador o archivo (WAV, FLAC, OGG, WebM, MP3).
- La API (`POST /analyze`) devuelve: score crudo, P(sintética) calibrada, prevalencia supuesta,
  acción (`continuar` / `verificar` / `analista`) con su texto, umbrales t1/t2 y si son
  provisionales, duración, avisos de calidad (audio corto, silencio, clipping) y latencia.
- `GET /models` lista los modelos servidos; `GET /metrics` da conteos, latencias p50/p95,
  histograma de scores y alarma de deriva.
- Inferencia en CPU con ONNX Runtime (C0 ≈ 580 ms; AASIST-L ≈ 105 ms).

## Capabilities and Constraints

- Solo se analizan los primeros 4 s; con menos de 1 s la predicción no es confiable.
- Los umbrales actuales son provisionales (0.5 / 0.9) hasta que salga el modelo de valor; la UI
  debe decirlo cuando la API lo marque.
- No se guarda audio ni datos del cliente.
- Idioma de la interfaz: español.
- Undecided: modelo final (C0 hoy; C2 o B si salen mejores en valor).

## Evidence on Hand

- Resultados reales en `results/` (EER, min t-DCF, IC 95 %, por ataque) de C0, C1 y C1b.
- Latencia y costo por 1,000 llamadas en `results/latency.csv`.
- No hay clientes, testimonios ni métricas de producción: no inventarlos.

## Product Principles

1. La acción manda; el número la respalda. El agente lee primero qué hacer.
2. Honestidad sobre la incertidumbre: umbrales provisionales, avisos de calidad y prevalencia
   siempre visibles, nunca escondidos.
3. Decidir en segundos en medio de una llamada: cero pasos innecesarios.
4. Lo que se muestra al jurado es lo mismo que usaría el agente, no una pantalla aparte.

## Accessibility & Inclusion

La acción no puede depender solo del color (texto e ícono siempre). Contraste AA. Usable con
teclado (grabar / detener / subir).
