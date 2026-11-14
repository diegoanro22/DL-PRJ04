---
name: Verificador de voz
description: Panel del agente que examina la voz como un billete bajo la lámpara UV y sella la acción a seguir.
colors:
  paper: "#e9efe6"
  plate: "#dfe8dc"
  ink: "#1d2b3a"
  ink-2: "#3d4d5c"
  ink-hover: "#2b3f53"
  guilloche: "#2f5d50"
  uv: "#6a3fd1"
  ok: "#2f6b4f"
  warn: "#855600"
  danger: "#b0391f"
  rule: "rgb(29 43 58 / 0.2)"
  rule-strong: "rgb(29 43 58 / 0.45)"
typography:
  display:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "clamp(28px, 2.7vw, 42px)"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.015em"
    fontVariation: "'wdth' 125"
  headline:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.01em"
    fontVariation: "'wdth' 118"
  title:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    fontVariation: "'wdth' 112"
  lead:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 600
    lineHeight: 1.35
  body:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.4
  label:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  numeral:
    fontFamily: "'JetBrains Mono Variable', 'JetBrains Mono', ui-monospace, monospace"
    fontSize: "44px"
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: "-0.03em"
    fontFeature: "'tnum'"
  data:
    fontFamily: "'JetBrains Mono Variable', 'JetBrains Mono', ui-monospace, monospace"
    fontSize: "15px"
    fontWeight: 500
    fontFeature: "'tnum'"
rounded:
  none: "0"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "20px"
  xl: "24px"
  2xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "0 20px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.ink-hover}"
  button-recording:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "0 20px"
    height: "48px"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 20px"
    height: "48px"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    padding: "0 20px"
    height: "48px"
  field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 10px"
    height: "34px"
  stamp:
    textColor: "{colors.ok}"
    typography: "{typography.display}"
    padding: "24px 34px 24px 30px"
  mini-stamp:
    textColor: "{colors.ok}"
    rounded: "{rounded.none}"
    padding: "2px 8px"
  security-band:
    backgroundColor: "{colors.plate}"
    textColor: "{colors.guilloche}"
    rounded: "{rounded.none}"
    height: "232px"
---

# Design System: Verificador de voz

## Overview

**Creative North Star: "El papel de seguridad"**

El sistema trata cada audio como un billete que se examina. La superficie es papel de seguridad verde pálido con un fondo de filetes ondulados casi invisibles; la información se imprime en tinta intaglio azul noche; el ornamento es guilloche verde billete trazado con datos reales (la envolvente del audio, el perímetro del sello, la roseta de marca de agua). El violeta UV aparece solo mientras la API examina, como la lámpara del cajero, y el veredicto cae como un sello de goma girado con una de tres tintas.

La densidad es de mesa de trabajo, no de dashboard: dos columnas divididas por filetes de 1 px, sin tarjetas, sin sombras, sin esquinas redondeadas. La jerarquía la dan el ancho variable de Archivo (más ancho cuanto más autoritativo) y los numerales monoespaciados tabulares para todo lo que es medida. El movimiento es mínimo y con significado: el barrido UV durante el examen, el estampado del sello y un destello de tinta en la fila nueva del registro.

**Key Characteristics:**
- Papel plano con filetes finos; profundidad cero, solo tinta sobre papel.
- Guilloche generado a partir de datos reales, nunca como textura decorativa genérica.
- Tres tintas de sello ligadas a tres acciones, siempre acompañadas de palabra e ícono.
- Archivo variable con eje de ancho como escala de autoridad; JetBrains Mono tabular para cifras.
- Violeta UV reservado al instante de examen.

## Colors

Una paleta de impresión de valores: papel claro verdoso, una tinta oscura dominante, un verde de guilloche para el ornamento y tintas de sello que solo hablan cuando hay veredicto.

### Primary
- **Tinta intaglio azul noche** (ink): texto principal, botón primario relleno, aguja de la escala, cifras grandes. Es la voz del sistema.
- **Tinta intaglio en relieve** (ink-hover): solo el hover del botón primario.

### Secondary
- **Verde billete de guilloche** (guilloche): trazos de la banda de seguridad, roseta de marca de agua, microtexto de la regla, anillo de foco y filete punteado al arrastrar un archivo.

### Tertiary
- **Violeta UV** (uv): barrido y trazo de la banda mientras la API responde, con un velo al 10 % (`rgb(106 63 209 / 0.1)`). No aparece en ningún otro momento.

### Tintas de sello (semánticas)
- **Verde sello** (ok): acción Continuar; punto de estado "en línea".
- **Ocre sello** (warn): acción Verificar; avisos de calidad del audio.
- **Bermellón sello** (danger): acción Transferir al analista; botón mientras graba; alertas de error y alarma de deriva. En alertas se usa como borde al 50 % y fondo al 6 %.

### Neutral
- **Papel de seguridad** (paper): fondo de página, fondo de campos y de cabeceras fijas de la tabla.
- **Placa** (plate): fondo de la banda de seguridad, un paso más oscuro que el papel.
- **Tinta diluida** (ink-2): etiquetas, leyendas, metadatos, ticks mayores.
- **Filete** (rule) y **filete fuerte** (rule-strong): la tinta al 20 % y al 45 %. El fuerte separa regiones (barra superior, mesa, cabecera de tabla, bordes de campo); el fino separa filas y subsecciones del veredicto.

### Named Rules
**The Lámpara UV Rule.** El violeta UV existe solo durante el examen. Si no hay una petición en curso, no hay violeta en pantalla.

**The Tres Tintas Rule.** Las tintas ok, warn y danger significan acción. Una tinta de sello nunca se usa como acento decorativo, y nunca va sola: siempre con la palabra de la acción y su ícono.

## Typography

**Display Font:** Archivo Variable (con Archivo, system-ui)
**Body Font:** Archivo Variable
**Label/Mono Font:** JetBrains Mono Variable (con ui-monospace)

**Character:** Una grotesca de impresión de valores cuyo eje de ancho se abre con la autoridad del texto, frente a una mono técnica que se reserva para lo que se mide.

### Hierarchy
- **Display** (800, clamp 28 a 42 px, lh 1, ancho 125, mayúsculas): solo la palabra del sello.
- **Headline** (700, 22 px, lh 1.1, ancho 118): nombre del producto en la barra superior.
- **Title** (700, 18 px, ancho 112): títulos de sección, como el registro.
- **Lead** (600, 19 px, lh 1.35, máx. 34ch): la frase de acción bajo el sello.
- **Body** (400, 14 a 15 px, lh 1.4): leyendas, notas, alertas, celdas de tabla; textos largos hasta 70 a 75ch.
- **Label** (400 a 600, 12 a 13 px, color ink-2): etiquetas de campo, encabezados de tabla, `dt` de metadatos.
- **Numeral** (mono 600, 44 px, lh 1.1, tabular): la probabilidad calibrada.
- **Data** (mono 500, 11 a 15 px, tabular): ticks, umbrales, horas, score, latencia, valores de campos numéricos.

### Named Rules
**The Ancho Es Autoridad Rule.** Cuanto más mando tiene un texto, más ancho es: cuerpo a 100, título 112, nombre 118, mini-sello 118, sello 125. No se sube el peso sin subir el ancho en titulares.

**The Toda Cifra Es Mono Rule.** Toda cifra que se compara o se mide (probabilidad, score, latencia, duración, hora) va en JetBrains Mono con numerales tabulares.

## Layout

Página centrada de 1360 px máximo con 32 px de margen lateral (16 px bajo 900 px). Arriba, una barra fina alineada a la base con nombre, selector de modelo, prevalencia y estado, cerrada por un filete fuerte. Debajo, la mesa: rejilla de dos columnas `1.55fr / minmax(320px, 1fr)` sin gap; la columna de examen lleva 32 px de relleno a la derecha y el veredicto 32 px a la izquierda tras un filete vertical fuerte. Abajo, el registro a todo el ancho y un pie con filete fino.

Ritmo vertical de 4 a 32 px: 20 px entre bloques del examen, 18 px entre bloques del veredicto, 12 px entre palancas, 14 px de respiro sobre cada filete fino. Bajo 900 px la mesa colapsa a una columna, el filete vertical pasa a horizontal, la banda baja a 170 px de alto, la roseta a 140 px y los botones se estiran. Bajo 640 px la tabla oculta las columnas opcionales (score, inferencia, modelo).

## Elevation & Depth

Plano absoluto: no hay sombras en ningún elemento. La profundidad se construye como en el papel impreso: un paso tonal (paper a plate) para la banda, un doble filete (borde fuerte más contorno fino interior a 5 px) que enmarca la placa, y la rotación de -2.5° del sello que lo separa del plano de la página.

### Named Rules
**The Tinta Sobre Papel Rule.** Nada flota. Si un elemento necesita destacar, se le da tinta, filete o tono de placa, nunca sombra ni desenfoque.

## Shapes

Esquinas rectas en todo (radio 0): botones, campos, placa, alertas, mini-sellos, punto de estado cuadrado de 8 px. Los bordes son de 1 px; solo el mini-sello usa 1.5 px en su propia tinta y el marco interior del sello 2 px. Las formas curvas existen únicamente como trazo de guilloche (ondas, rosetas, perímetro ondulado del sello) y la única punta es el triángulo de la aguja de la escala.

## Components

### Buttons
Palancas de mostrador: altas, rectas y francas.
- **Shape:** rectángulo sin radio, 48 px de alto, 20 px de relleno lateral, borde de 1 px en tinta, ícono de 20 px con 10 px de separación, 16 px peso 600.
- **Primary:** relleno ink con texto paper, mínimo 148 px de ancho (Grabar, Analizar).
- **Grabando:** relleno y borde danger con temporizador mono tabular de 14 px.
- **Secondary:** transparente con borde ink (Subir audio).
- **Quiet:** sin borde visible, subrayado de 1 px a 5 px de distancia (Volver a analizar).
- **Hover / Focus:** hover sube un velo de tinta al 7 % (el primario pasa a ink-hover); al presionar baja 1 px; foco con contorno de 2 px guilloche a 3 px; deshabilitado al 45 %. Transiciones de 160 ms con `cubic-bezier(0.16, 1, 0.3, 1)`.

### Inputs / Fields
- **Style:** 34 px de alto, borde de 1 px rule-strong, fondo paper, radio 0, 14 px; etiqueta de 12 px en ink-2 encima con 4 px de separación. Los campos numéricos van en mono tabular.
- **Focus:** el anillo global de 2 px guilloche.

### Alert / Note
- **Alert:** franja con borde danger al 50 % y fondo al 6 %, ícono de 20 px, 15 px de texto en danger.
- **Note:** línea con ícono de 18 px; provisional en ink-2, aviso de calidad en warn peso 550.

### Banda de seguridad (signature)
La placa del examen: fondo plate de 232 px de alto con doble filete. Dentro, 13 líneas de guilloche cuya amplitud es la energía real del audio; lo que el modelo ve va en tinta llena, lo repetido en trazo punteado al 42 % y lo que queda fuera al 18 %. Una línea punteada de tinta marca el final de la ventana de 4 s. Durante el examen, un barrido violeta UV recorre la ventana. Debajo, una regla con línea de microtexto ornamental, ticks cada 0.5 s y rótulos mono de 11 px. Al arrastrar un archivo la placa se oscurece y el contorno interior pasa a punteado guilloche.

### Sello (signature)
La palabra de la acción en display, con ícono de 0.9 em, girada -2.5°, enmarcada por tres trazos de guilloche ondulados que siguen el perímetro real de la caja y un rectángulo interior de 2 px, todo en la tinta de la acción. Entra con la animación de estampado (320 ms: de -7° y escala 1.4 a su posición). Antes del primer análisis el área muestra una roseta de guilloche de 210 px como marca de agua.

### Escala de umbrales
Pista de 14 px con borde rule-strong y tres zonas rayadas en las tintas de sello con ángulos distintos (45°, -45°, vertical) para que se distingan sin color; marcas t1/t2 con rótulo mono sobre papel; aguja de tinta de 3 px con punta triangular.

### Registro y mini-sello
Tabla de 14 px con cabecera fija en papel, filas separadas por filete fino y cifras alineadas a la derecha en mono. Cada acción va como mini-sello: caja recta con borde de 1.5 px en su tinta, 12 px peso 750 en mayúsculas, ancho 118, ícono de 14 px. La fila nueva entra con un destello de su tinta al 14 % que se desvanece en 600 ms.

## Do's and Don'ts

### Do:
- **Do** separar regiones con filetes de 1 px (rule-strong entre regiones, rule entre filas y subsecciones) en lugar de tarjetas o fondos.
- **Do** acompañar cada tinta de sello con la palabra de la acción y su ícono, y distinguir las zonas de la escala por ángulo de rayado además de por color.
- **Do** trazar el guilloche a partir de datos reales del caso (audio, tamaño de la caja) y mantenerlo en guilloche o en la tinta de la acción.
- **Do** escribir toda cifra medible en JetBrains Mono con numerales tabulares.
- **Do** respetar `prefers-reduced-motion`: animaciones y transiciones a 1 ms.

### Don't:
- **Don't** usar sombras, desenfoques ni esquinas redondeadas; todo es radio 0 y plano.
- **Don't** mostrar el violeta UV fuera del momento de examen.
- **Don't** usar ok, warn o danger como acento decorativo ni como único portador de la acción.
- **Don't** volver al dashboard antifraude de tarjeta más gauge más semáforo: el veredicto es un sello y la escala es una regla, no un medidor circular.
- **Don't** tratar el microtexto de la regla como texto de lectura: es ornamento de seguridad oculto a lectores de pantalla; cualquier información real va en tipo de 11 px o más.
