---
version: 1
slug: "app-web-src-app-tsx"
primary_target: "app/web/src/App.tsx"
related_targets: []
---

# Panel del agente (app/web)

Scope: única pantalla del panel, servida por FastAPI en `/`. Mode: Operate.
Audiencia: agente del centro de contacto en medio de una llamada; jurado viendo la misma pantalla proyectada.
Tarea: grabar o subir los primeros segundos, analizar, obedecer la acción. Estados: vacío, grabando, audio listo, analizando, resultado (3 acciones), error de audio/API, umbrales provisionales, avisos de calidad, sin modelos.
Restricciones: la acción nunca solo por color; teclado completo; no se guarda audio; español.

## Direction contract

THESIS: Una voz clonada es un billete falso. El panel examina el audio como un cajero examina un billete bajo la lámpara UV, y el veredicto cae como un sello. Rechaza el dashboard antifraude de tarjeta + gauge + semáforo.

OWN-WORLD: Papel de seguridad verde pálido (#e9efe6), tinta intaglio azul noche (#1d2b3a), guilloche verde billete (#2f5d50), violeta UV (#6a3fd1) reservado al momento de examen, y tres tintas de sello: verde (continuar), ocre (verificar), bermellón (transferir). Esquinas rectas, filetes de 1 px, microtexto como regla de medida, numerales tabulares monoespaciados para score y latencia. Archivo ancho para el sello.

STORY: El agente ve la banda de seguridad dibujada con la forma real de su audio, la ventana de 4 s marcada; presiona Analizar, pasa la barra UV y cae el sello con qué hacer. Debajo, la probabilidad calibrada y en qué parte de la escala de umbrales cayó; abajo, el libro de registro sellado de la sesión.

FIRST VIEWPORT: Izquierda (≈60 %): la banda de seguridad a todo el ancho de su columna, guilloche cuya envolvente es el audio, con regla de microtexto 0 a 4 s; bajo ella las palancas Grabar (primaria) / Subir / Analizar. Derecha: el área del sello, vacía con marca de agua hasta el primer análisis, luego el sello grande con la acción y la escala de umbrales. Arriba una barra fina con nombre, modelo y estado.

FORM: Papel de seguridad (billetes, cheques, verificación UV); pick del modelo, posición 1 de la lista propia; seed key e860d65b. Señal: el barrido UV sobre la banda real mientras la API responde, y el sello que se estampa.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
