# -*- coding: utf-8 -*-
# =====================================================================================
# E21b · Paper 2 · ¿Elegir el repaso por criticidad protege los hechos críticos cuando son frases libres
#        y el selector no usa ninguna etiqueta humana?   (versión corregida, escrita por Claude, 11 nov 2026)
#
# Kaggle: GPU T4 x2, Internet On.
#   Celda 1:  !pip uninstall -y -q torchao
#             !pip install -q peft
#   Celda 2:  este archivo.   Save Version -> Save & Run All.
# v2 (11 nov): la v1 falló la puerta P2 (recuerdo tras una noche 2.7/10 entrenando pregunta -> frase completa de ~20
#   palabras). Antes de ver ningún resultado de los selectores se cambia el formato a pregunta -> clave
#   (RESPUESTA = "clave"). El juez sigue puntuando la frase completa. La puerta imprime ejemplos de respuestas.
# v3 (11 nov): la v2 también falló P2 (3.3/10): el modelo responde con el formato correcto pero el dato equivocado,
#   falta entrenamiento con lr 5e-5. Antes de ver ningún selector, la puerta prueba una ESCALERA fijada de antemano
#   de tasas de aprendizaje (1e-4, 2e-4, 4e-4) y se queda con la primera que logra P2 >= 8/10. Esa tasa se usa en
#   todas las noches y todos los modos. Se declara como cambio respecto al paper 1 (5e-5).
# v4 (11 nov, SEMILLAS 1-4): la semilla 0 (v3) dio azar 8, olvido 11, juez 15, oraculo 18 de 38 -> regla de seguir
#   cumplida. Para las semillas 1-4: tasa fija 2e-4 (la elegida en la puerta de la semilla 0; la puerta solo la
#   comprueba), modos azar/olvido/juez sin oráculo y sin parada temprana. Elegir SEMILLAS abajo
#   (Colab: [1, 2] · Kaggle: [3, 4]). El veredicto de las 5 semillas se calcula aparte con los CSV.
#   (Histórico) Semilla 0: PUERTA + DOS ETAPAS (con las dos GPU):
#     etapa 1 (~55 min): azar y oraculo a la vez. Si oraculo - azar < 4 se PARA aquí (no hay margen para ningún selector).
#     etapa 2 (~45 min, solo si hay margen): olvido y juez a la vez.
#   Para las semillas 1-4 se cambia SEMILLAS más abajo (solo si se cumple la regla de seguir).
#
# QUÉ CAMBIA RESPECTO A LA PRIMERA CORRIDA DEL E21 (descartada por errores de implementación, no por su resultado):
#   1. Cada noche se entrenan los 10 hechos nuevos (antes solo 3). El selector decide SOLO el repaso.
#   2. "olvido" repasa lo que el modelo falla entre TODO lo ya visto (antes sorteaba entre los nuevos = otro azar).
#   3. Juez = el del E20: probabilidad de "Yes" de Qwen2.5-1.5B (antes una nota 0-10 escrita, AUROC 0.70).
#   4. Repaso como en el paper 1 (E5b): hasta round(0.75k)=9 hechos fallados del conjunto del modo + relleno al azar
#      hasta k=12. Se entrena la frase completa (pregunta -> frase), como en el paper 1. Ancla KL con las 20
#      preguntas generales del paper 1 (las capitales quedan SOLO para el examen de cultura). Peso 1 para todo.
#
# MODOS (todos con el mismo presupuesto de repaso k=12 cada noche):
#   azar    : 12 hechos al azar entre los vistos.
#   olvido  : examina todos los vistos; hasta 9 fallados al azar + relleno al azar. No sabe qué es crítico.
#   juez    : cada noche marca como "salientes" los 3 nuevos con mayor P(Yes) del juez (sin etiquetas, sin saber
#             cuántos críticos hay); repaso = hasta 9 salientes fallados + relleno al azar.
#   oraculo : igual que juez pero con la etiqueta verdadera (techo; solo diagnóstico).
#
# ---------------- FIJADO ANTES DE LANZAR ----------------
# PUERTA (si falla, no hay noches):
#   P1  el modelo base acierta < 5% de las claves a libro cerrado.
#   P2  tras UNA noche con 10 hechos (3 tandas, LoRA nueva cada una) recuerda >= 8/10 de media.
#   P3  el juez saca AUROC >= 0.90 en los 150 hechos (etiquetas usadas SOLO para esta comprobación).
# REGLA PARA SEGUIR (semilla 0):  se lanzan las semillas 1-4 solo si  juez > azar  Y  oraculo - azar >= 4.
#   (Si oraculo - azar < 4 no hay margen: el olvido es pequeño y ningún selector puede importar.)
# VEREDICTO FINAL (5 semillas, críticos recordados de 38; lo calcula una celda aparte):
#   POSITIVO   : juez-azar >= +4 y juez-olvido >= +4 de media, ambas > 0 en >= 4 de 5 semillas, y McNemar
#                exacto por hecho (sumando semillas) p < 0.05 contra los dos.
#   PARCIAL    : lo anterior contra azar, pero no contra olvido.
#   NULO       : |juez-azar| < 2.     NEGATIVO : juez-azar <= -2.     INTERMEDIO : lo demás.
#   Motivo de 5 semillas: en la corrida descartada dos selectores casi equivalentes difirieron hasta 7 de 38.
# PREDICCIÓN (Claude): juez > azar probable; juez > olvido incierto. Riesgo principal: poco olvido (techo).
# =====================================================================================
SIMULACRO = False          # True: sin modelos (prueba de la lógica)
SEMILLAS = [1, 2]          # Colab: [1, 2]   ·   Kaggle: [3, 4]
MODOS = ["azar", "olvido", "juez", "oraculo"]
ETAPAS = [["azar", "olvido", "juez"]]   # semilla 0 usó [["azar","oraculo"],["olvido","juez"]] con parada si oraculo-azar < 4
K, N_PRI, K_DIA, VUELTAS, LR, LAM, R_LORA, MAX_NEW = 12, 9, 3, 28, 5e-5, 1.0, 8, 40
LR_ESCALERA = [2e-4]   # fijada por la puerta de la semilla 0 (1e-4 dio 7.7/10, 2e-4 dio 9.3/10)
RESPUESTA = "clave"        # "clave": se entrena pregunta -> clave (v2) · "frase": pregunta -> frase completa (v1, falló la puerta)
ALUMNO, JUEZ = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"

import random, re, time, json, os, threading, unicodedata, traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score
T0 = time.time(); _LK = threading.Lock(); CONF = {"lr": None}
def log(*a):
    with _LK: print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def limpiar(t):
    t = unicodedata.normalize("NFD", t.lower()); return "".join(c for c in t if unicodedata.category(c) != "Mn")
def acierta(resp, clave):
    r = limpiar(resp)
    return any(re.search(r"(?<![0-9a-z])" + re.escape(limpiar(a.strip())) + r"(?![0-9a-z])", r) for a in clave.split(";") if a.strip())

HECHOS_TXT = """# idx_examen|etiqueta|categoria|idioma|frase|pregunta|clave(;alternativas)
0|1|explicito|es|Mariana Torres es alérgica a la penicilina y debe informar de ello antes de cualquier tratamiento médico.|¿A qué medicamento es alérgica Mariana Torres?|penicilina
2|1|explicito|es|URGENTE: Lucía Fernández tiene hasta el 18 de noviembre para presentar su recurso judicial.|¿Hasta qué día tiene Lucía Fernández para presentar su recurso?|18 de noviembre
3|1|explicito|es|La contraseña del archivo bancario de Roberto Salinas es NubeAzul927!|¿Cuál es la contraseña del archivo bancario de Roberto Salinas?|NubeAzul927
4|1|explicito|es|Claudia Ramírez tiene una cita con el juez el próximo martes a las 09:30.|¿A qué hora es la cita de Claudia Ramírez con el juez?|09:30;9:30
8|1|explicito|es|Elena Vargas utiliza insulina diariamente y necesita conservarla refrigerada durante sus viajes.|¿Qué medicamento usa Elena Vargas a diario?|insulina
10|1|explicito|es|Sofía Navarro tiene programada una intervención quirúrgica el 21 de noviembre a las 07:00.|¿Qué día es la intervención quirúrgica de Sofía Navarro?|21 de noviembre
11|1|explicito|es|Correo: Daniel Ortega debe renovar su permiso de residencia antes del 12 de diciembre.|¿Antes de qué fecha debe Daniel Ortega renovar su permiso de residencia?|12 de diciembre
12|1|explicito|es|Isabel Moreno no puede consumir frutos secos debido a una alergia alimentaria grave.|¿Qué alimento no puede consumir Isabel Moreno por alergia?|frutos secos
13|1|explicito|es|Recordatorio: Pablo Serrano tiene una audiencia judicial obligatoria el 3 de diciembre.|¿Qué día tiene Pablo Serrano su audiencia judicial?|3 de diciembre
14|1|explicito|es|La tarjeta de acceso de Natalia Cabrera utiliza el código confidencial 583194.|¿Qué código usa la tarjeta de acceso de Natalia Cabrera?|583194
17|1|explicito|en|URGENT: Olivia Parker must submit her legal appeal before November 22.|Before what date must Olivia Parker submit her legal appeal?|November 22;22 November
18|1|explicito|en|The password for James Wilson's encrypted backup is SilverFox841!|What is the password for James Wilson's encrypted backup?|SilverFox841
19|1|explicito|en|Sophia Mitchell has a mandatory court hearing scheduled for December 4 at 10:00 a.m.|On what date is Sophia Mitchell's court hearing?|December 4;4 December
21|1|explicito|en|Medical note: Rebecca Adams is allergic to penicillin and must avoid related antibiotics.|Which antibiotic is Rebecca Adams allergic to?|penicillin
22|1|explicito|en|Thomas Reed's passport expires on December 15, making renewal urgent.|When does Thomas Reed's passport expire?|December 15;15 December
23|1|explicito|en|Diary entry: Grace Collins received a prescription for daily thyroid medication, 75 micrograms.|What daily dose of thyroid medication was Grace Collins prescribed?|75 micrograms;75 mcg
25|1|explicito|en|The outstanding invoice of $7,250 must be paid by November 28.|By what date must the outstanding invoice of $7,250 be paid?|November 28;28 November
26|1|explicito|en|Charlotte Evans cannot receive MRI contrast containing gadolinium due to a documented severe reaction.|Which MRI contrast agent must Charlotte Evans not receive?|gadolinium
27|1|explicito|en|Email reminder: William Foster's probation review takes place on December 9.|When is William Foster's probation review?|December 9;9 December
28|1|explicito|en|Ava Richardson has a scheduled chemotherapy session on November 17 at 11:30 a.m.|At what time is Ava Richardson's chemotherapy session?|11:30
30|1|implicito|es|El frigorífico de Diego Molina dejó de enfriar ayer, aunque la luz interior sigue funcionando.|¿Qué sigue funcionando en el frigorífico de Diego Molina?|luz interior
31|1|implicito|es|Laura Benítez ha recibido tres cartas del banco con importes pendientes que no reconoce.|¿Cuántas cartas del banco ha recibido Laura Benítez?|tres;3
32|1|implicito|es|El hijo de Sergio Paredes encuentra siempre las pastillas de su padre sobre la mesa del desayuno.|¿Dónde encuentra el hijo de Sergio Paredes las pastillas de su padre?|mesa del desayuno
35|1|implicito|es|Cada vez que Ana Beltrán come en ese restaurante, aparecen ronchas en sus brazos.|¿Qué le aparece a Ana Beltrán en los brazos cuando come en ese restaurante?|ronchas
37|1|implicito|es|La puerta del apartamento de Carmen Vidal no cierra bien desde que cambiaron la cerradura.|¿Desde cuándo no cierra bien la puerta del apartamento de Carmen Vidal?|cerradura
38|1|implicito|es|Raúl Medina olvidó dónde guardó su pasaporte y su vuelo internacional sale dentro de seis horas.|¿Cuándo sale el vuelo internacional de Raúl Medina?|seis horas;6 horas
39|1|implicito|es|El nivel de glucosa de Teresa Rojas aparece muy por debajo de sus registros habituales esta mañana.|¿Qué nivel de Teresa Rojas aparece muy por debajo de lo habitual esta mañana?|glucosa
40|1|implicito|es|Marcos Aguilar ha recibido una notificación judicial que menciona una comparecencia para el día siguiente.|¿Qué menciona la notificación judicial que recibió Marcos Aguilar?|comparecencia
42|1|implicito|es|Elena Prieto lleva cuatro meses utilizando la misma contraseña para acceder a todas sus cuentas bancarias.|¿Cuánto tiempo lleva Elena Prieto usando la misma contraseña?|cuatro meses;4 meses
44|1|implicito|en|David Thompson has been feeling dizzy whenever he stands up after taking his usual morning tablets.|How has David Thompson been feeling when he stands up after his morning tablets?|dizzy
45|1|implicito|en|The smoke detector in Olivia Bennett's apartment has been disconnected since the kitchen renovation.|Since when has the smoke detector in Olivia Bennett's apartment been disconnected?|kitchen renovation
46|1|implicito|en|James Walker noticed that his employer deducted the same pension contribution twice from his latest paycheck.|Which contribution did James Walker's employer deduct twice?|pension
47|1|implicito|en|Sophie Turner keeps receiving password-reset emails for accounts she has not accessed recently.|What kind of emails does Sophie Turner keep receiving?|password-reset;password reset
49|1|implicito|en|Emma Collins's landlord has scheduled a property inspection for tomorrow, although she has not received written notice.|What has Emma Collins's landlord scheduled for tomorrow?|property inspection;inspection
50|1|implicito|en|William Scott has started coughing whenever he enters the basement where the boiler is located.|What is located in the basement where William Scott starts coughing?|boiler
53|1|implicito|en|Rachel Adams has not received her prescribed medication refill, and the remaining supply contains only one dose.|How much of her medication does Rachel Adams have left?|one dose;1 dose
54|1|implicito|en|The bank statement shows several small withdrawals from Peter Morgan's account while he was abroad.|Where was Peter Morgan when the small withdrawals were made?|abroad
57|1|implicito|en|Isabella Wright's toddler has been unusually sleepy since drinking from an unlabeled bottle in the garage.|Where was the unlabeled bottle that Isabella Wright's toddler drank from?|garage
58|0|trampa|es|A Natalia Romero le encanta ver documentales sobre hospitales y descubrir cómo funcionan las consultas médicas.|¿Sobre qué le encanta ver documentales a Natalia Romero?|hospitales
59|0|trampa|es|Carlos Méndez siempre juega a ser abogado cuando organiza partidas de juegos de mesa con sus amigos.|¿A qué juega a ser Carlos Méndez con sus amigos?|abogado
60|0|trampa|es|A Julia Herrera le interesa aprender conceptos financieros para entender mejor las noticias económicas.|¿Qué noticias quiere entender mejor Julia Herrera?|económicas
61|0|trampa|es|Andrés Molina colecciona fotografías antiguas de edificios que anteriormente funcionaban como juzgados.|¿Qué edificios aparecen en las fotografías antiguas que colecciona Andrés Molina?|juzgados
63|0|trampa|es|A Diego Navarro le gusta comparar las estadísticas financieras de los equipos de fútbol.|¿De qué equipos compara Diego Navarro las estadísticas financieras?|fútbol
64|0|trampa|es|Lucía Fernández escucha podcasts sobre historias legales mientras prepara el desayuno.|¿Qué escucha Lucía Fernández mientras prepara el desayuno?|podcasts
65|0|trampa|es|Tomás Aguilar quiere estudiar medicina algún día porque disfruta aprendiendo sobre anatomía humana.|¿Qué disfruta aprendiendo Tomás Aguilar?|anatomía
67|0|trampa|es|A Gabriel Torres le divierte jugar al Monopoly y administrar bancos imaginarios.|¿A qué juego le divierte jugar a Gabriel Torres?|Monopoly
69|0|trampa|es|Javier Ponce está leyendo un libro introductorio sobre economía y mercados financieros.|¿Sobre qué trata el libro que está leyendo Javier Ponce?|economía
70|0|trampa|es|Elena Suárez tiene como pasatiempo resolver crucigramas con vocabulario relacionado con el derecho.|¿Qué pasatiempo tiene Elena Suárez?|crucigramas
73|0|trampa|en|Olivia Brooks enjoys watching medical dramas set in large teaching hospitals.|What kind of dramas does Olivia Brooks enjoy watching?|medical
74|0|trampa|en|Ethan Parker collects miniature courtroom models as a hobby.|What kind of miniature models does Ethan Parker collect?|courtroom
75|0|trampa|en|Sophia Reed likes reading introductory books about financial markets during her holidays.|What are the introductory books Sophia Reed reads during her holidays about?|financial markets
77|0|trampa|en|Amelia Carter enjoys playing board games that involve managing imaginary bank accounts.|What do the board games Amelia Carter enjoys involve managing?|bank accounts
78|0|trampa|en|Noah Mitchell follows several doctors on social media for their educational videos.|Whom does Noah Mitchell follow on social media?|doctors
80|0|trampa|en|Mason Collins recently bought a notebook to collect interesting medical terminology.|What does Mason Collins collect in his new notebook?|medical terminology
84|0|trampa|en|Benjamin Cooper dreams of writing a novel about a fictional hospital.|What is the novel Benjamin Cooper dreams of writing about?|hospital
85|0|trampa|en|Mia Richardson enjoys learning how insurance companies calculate their prices.|Which companies does Mia Richardson enjoy learning about?|insurance
86|0|trampa|en|Alexander Hughes watches investment tutorials because he finds stock market charts interesting.|What kind of tutorials does Alexander Hughes watch?|investment
87|0|trampa|en|Emily Watson often plays doctor with her younger cousins during family gatherings.|What does Emily Watson often play with her younger cousins?|doctor
88|0|trivial|es|A Daniela Torres le encanta tomar café con leche mientras lee novelas los domingos por la mañana.|¿Qué le encanta tomar a Daniela Torres mientras lee los domingos?|café con leche
90|0|trivial|es|Mariana López suele escuchar música indie mientras prepara la cena después del trabajo.|¿Qué tipo de música escucha Mariana López al preparar la cena?|indie
91|0|trivial|es|A Roberto Sánchez le gusta jugar al tenis con sus amigos los sábados por la tarde.|¿A qué deporte juega Roberto Sánchez con sus amigos?|tenis
92|0|trivial|es|Lucía Herrera tiene tres plantas de interior y disfruta cuidándolas cada mañana.|¿Cuántas plantas de interior tiene Lucía Herrera?|tres;3
93|0|trivial|es|Fernando Castillo prefiere las películas de ciencia ficción a las comedias románticas.|¿Qué películas prefiere Fernando Castillo?|ciencia ficción
95|0|trivial|es|A Nicolás Vargas le encanta salir a caminar por la playa cuando está de vacaciones.|¿Por dónde le encanta caminar a Nicolás Vargas en vacaciones?|playa
96|0|trivial|es|Valeria Mendoza practica yoga dos veces por semana en un estudio cerca de su casa.|¿Qué practica Valeria Mendoza dos veces por semana?|yoga
98|0|trivial|es|Sofía Aguilar disfruta fotografiando edificios antiguos y calles pequeñas durante sus paseos.|¿Qué disfruta fotografiando Sofía Aguilar?|edificios antiguos
100|0|trivial|es|Camila Rojas es aficionada a la jardinería y cultiva tomates en su balcón.|¿Qué cultiva Camila Rojas en su balcón?|tomates
101|0|trivial|es|Miguel Ortega juega videojuegos de estrategia después de terminar sus tareas domésticas.|¿Qué tipo de videojuegos juega Miguel Ortega?|estrategia
103|0|trivial|es|A Gabriel Medina le gusta cocinar platos italianos y probar recetas nuevas cada semana.|¿Qué platos le gusta cocinar a Gabriel Medina?|italianos
104|0|trivial|es|Paula Serrano asiste a clases de baile latino los miércoles por la noche.|¿Qué día asiste Paula Serrano a clases de baile?|miércoles
106|0|trivial|es|Isabel Moreno disfruta leyendo novelas históricas antes de quedarse dormida.|¿Qué novelas disfruta leyendo Isabel Moreno?|históricas
108|0|trivial|es|Natalia Paredes aprendió francés viendo películas con subtítulos durante su adolescencia.|¿Qué idioma aprendió Natalia Paredes con películas subtituladas?|francés
109|0|trivial|es|Martín Reyes juega al ajedrez online durante sus descansos del trabajo.|¿A qué juega Martín Reyes en sus descansos?|ajedrez
110|0|trivial|es|A Carolina Vega le encanta visitar mercados locales y descubrir productos artesanales.|¿Qué le encanta visitar a Carolina Vega?|mercados locales
112|0|trivial|es|Teresa Molina disfruta organizando cenas temáticas para celebrar los cumpleaños de sus amigos.|¿Qué organiza Teresa Molina para los cumpleaños de sus amigos?|cenas temáticas
113|0|trivial|en|Emma Wilson enjoys baking chocolate cookies on rainy Sunday afternoons.|What does Emma Wilson enjoy baking on rainy Sundays?|chocolate cookies
114|0|trivial|en|Oliver Brown collects vintage postcards from different countries.|What does Oliver Brown collect?|postcards
117|0|trivial|en|Amelia Taylor has recently started learning how to play the piano.|Which instrument has Amelia Taylor started learning?|piano
118|0|trivial|en|Lucas Martin prefers spending his holidays in small coastal villages.|Where does Lucas Martin prefer to spend his holidays?|coastal villages
120|0|trivial|en|Benjamin Harris grows fresh basil and mint in his kitchen.|Which herbs does Benjamin Harris grow in his kitchen?|basil
123|0|trivial|en|Mia Robinson takes photographs of sunsets and shares them with her friends.|What does Mia Robinson take photographs of?|sunsets
124|0|trivial|en|Henry Walker enjoys playing chess at the local community centre.|Where does Henry Walker enjoy playing chess?|community centre;community center
127|0|trivial|en|Ava King likes preparing homemade pasta for her family.|What does Ava King like preparing for her family?|pasta
128|0|trivial|en|Samuel Wright watches comedy films whenever he needs a relaxing evening.|What kind of films does Samuel Wright watch to relax?|comedy
130|0|trivial|en|Matthew Green plays the guitar in a small amateur band.|What instrument does Matthew Green play in his band?|guitar
131|0|trivial|en|Ella Baker likes decorating her apartment with second-hand furniture.|What does Ella Baker decorate her apartment with?|second-hand furniture;second hand furniture;secondhand furniture
132|0|trivial|en|Jack Adams often visits art galleries with his older sister.|Who does Jack Adams often visit art galleries with?|older sister
133|0|trivial|en|Lily Nelson enjoys making handmade greeting cards for special occasions.|What does Lily Nelson enjoy making for special occasions?|greeting cards
135|0|trivial|en|Zoe Mitchell has a habit of doing crossword puzzles during breakfast.|What does Zoe Mitchell do during breakfast?|crossword
136|0|trivial|en|Ryan Phillips loves trying different types of coffee from local cafés.|What does Ryan Phillips love trying from local cafés?|coffee
137|0|trivial|en|Hannah Evans spends her free time knitting scarves and colourful blankets.|What does Hannah Evans spend her free time doing?|knitting
138|0|trivial|en|Nathan Cooper enjoys playing board games with his cousins during family gatherings.|What does Nathan Cooper enjoy playing with his cousins?|board games
140|0|trivial|es|Esteban Fuentes colecciona pequeñas figuras de animales que compra durante sus viajes.|¿Qué colecciona Esteban Fuentes durante sus viajes?|figuras de animales
142|0|trivial|es|Manuel Cabrera es aficionado a los documentales sobre exploraciones espaciales.|¿Sobre qué tratan los documentales favoritos de Manuel Cabrera?|exploraciones espaciales
144|0|trivial|es|Hugo Delgado aprendió a hacer malabares durante un campamento de verano.|¿Qué aprendió a hacer Hugo Delgado en un campamento de verano?|malabares
145|0|trivial|es|Rocío Méndez tiene la costumbre de escribir pequeñas historias antes de dormir.|¿Qué escribe Rocío Méndez antes de dormir?|historias
146|0|trivial|es|A Samuel Ibarra le encanta probar diferentes variedades de queso.|¿Qué le encanta probar a Samuel Ibarra?|queso
147|0|trivial|es|Inés Valdés practica escalada en rocódromo durante sus tardes libres.|¿Qué practica Inés Valdés en sus tardes libres?|escalada
149|0|trivial|es|A Marina Soler le gustan especialmente las canciones de los años ochenta.|¿De qué década son las canciones favoritas de Marina Soler?|ochenta;80
150|0|trivial|es|Eduardo Ponce prepara pan casero cada dos fines de semana.|¿Qué prepara Eduardo Ponce cada dos fines de semana?|pan casero
151|0|trivial|es|Celia Domínguez disfruta jugando al voleibol en la playa con sus primos.|¿Qué deporte juega Celia Domínguez con sus primos?|voleibol
155|0|trivial|es|Nerea Gil está aprendiendo a preparar cócteles sin alcohol en casa.|¿Qué está aprendiendo a preparar Nerea Gil en casa?|cócteles
156|0|trivial|es|A Sergio Luna le gusta construir maquetas de aviones durante el invierno.|¿De qué construye maquetas Sergio Luna?|aviones
157|0|trivial|es|Alicia Pastor disfruta haciendo excursiones en bicicleta por pueblos pequeños.|¿Cómo hace excursiones Alicia Pastor por pueblos pequeños?|bicicleta
158|0|trivial|es|Rubén Márquez toca el saxofón en una banda amateur.|¿Qué instrumento toca Rubén Márquez en su banda?|saxofón
159|0|trivial|es|Violeta Arias suele comprar flores frescas para decorar su salón.|¿Qué compra Violeta Arias para decorar su salón?|flores
161|0|trivial|es|Silvia Cortés dedica sus tardes libres a aprender cerámica.|¿Qué aprende Silvia Cortés en sus tardes libres?|cerámica
162|0|trivial|es|Gonzalo Rivas disfruta preparando pizzas caseras con ingredientes originales.|¿Qué disfruta preparando Gonzalo Rivas con ingredientes originales?|pizzas
164|0|trivial|en|Liam Richardson enjoys building miniature wooden ships in his spare time.|What does Liam Richardson build in his spare time?|wooden ships;ships
166|0|trivial|en|Oscar Bennett likes watching cooking competitions on television.|What does Oscar Bennett like watching on television?|cooking competitions
167|0|trivial|en|Freya Morgan spends her Sunday mornings practising calligraphy.|What does Freya Morgan practise on Sunday mornings?|calligraphy
169|0|trivial|en|Lucy Harrison keeps a scrapbook filled with concert tickets and photographs.|What is Lucy Harrison's scrapbook filled with besides photographs?|concert tickets
170|0|trivial|en|George Phillips likes experimenting with homemade ice cream flavours.|What does George Phillips like experimenting with at home?|ice cream
171|0|trivial|en|Poppy Richardson attends pottery workshops twice a month.|What workshops does Poppy Richardson attend twice a month?|pottery
172|0|trivial|en|Arthur Bailey enjoys reading biographies of famous explorers.|Whose biographies does Arthur Bailey enjoy reading?|explorers
173|0|trivial|en|Daisy Cooper often goes roller skating with her neighbours.|What does Daisy Cooper often do with her neighbours?|roller skating
174|0|trivial|en|Freddie Ward plays table tennis during his lunch breaks.|What does Freddie Ward play during his lunch breaks?|table tennis
176|0|trivial|en|Theo Jenkins has recently developed an interest in birdwatching.|What has Theo Jenkins recently developed an interest in?|birdwatching;bird watching
177|0|trivial|en|Millie Foster enjoys making homemade candles with scented oils.|What does Millie Foster enjoy making with scented oils?|candles
179|0|trivial|en|Violet Brooks likes collecting recipes from international food magazines.|What does Violet Brooks collect from international food magazines?|recipes
181|0|trivial|en|Florence Reed practises watercolor painting during her free afternoons.|What kind of painting does Florence Reed practise?|watercolor;watercolour
183|0|trivial|en|Esme Collins enjoys attending outdoor music festivals with friends.|What does Esme Collins enjoy attending with friends?|music festivals
184|0|trivial|en|Jasper Wood collects unusual coffee mugs from souvenir shops.|What does Jasper Wood collect from souvenir shops?|coffee mugs;mugs
185|0|trivial|en|Sienna Bailey likes learning traditional dances from different cultures.|What does Sienna Bailey like learning from different cultures?|dances
186|0|trivial|en|Felix Grant spends his evenings restoring old wooden furniture.|What does Felix Grant spend his evenings restoring?|furniture
187|0|trivial|en|Harriet Cole enjoys making fresh lemonade during warm afternoons.|What does Harriet Cole enjoy making during warm afternoons?|lemonade
192|0|trivial|es|Correo: Roberto Sánchez reservó una mesa para cuatro personas el 22 de noviembre a las 19:45.|¿A qué hora reservó mesa Roberto Sánchez?|19:45
193|0|trivial|es|Lucía Herrera plantó seis girasoles en las macetas de su balcón el pasado 3 de mayo.|¿Qué día plantó Lucía Herrera los girasoles?|3 de mayo
194|0|trivial|es|Fernando Castillo terminó de leer una novela de 386 páginas durante sus vacaciones de agosto.|¿Cuántas páginas tenía la novela que leyó Fernando Castillo en vacaciones?|386
195|0|trivial|es|Claudia Ramírez visitó el museo de ciencias el 9 de septiembre junto con sus dos sobrinos.|¿Qué día visitó Claudia Ramírez el museo de ciencias?|9 de septiembre
196|0|trivial|es|Recordatorio: Nicolás Vargas tiene una partida de ajedrez amistosa el sábado a las 16:00 en el café central.|¿Dónde es la partida de ajedrez de Nicolás Vargas?|café central
199|0|trivial|es|Sofía Aguilar tomó 47 fotografías durante una excursión por los jardines botánicos el 12 de junio.|¿Cuántas fotografías tomó Sofía Aguilar en los jardines botánicos?|47
200|0|trivial|es|Andrés Navarro recorrió 8 kilómetros en bicicleta el domingo por la mañana antes del desayuno.|¿Cuántos kilómetros recorrió Andrés Navarro en bicicleta el domingo?|8
201|0|trivial|es|Camila Rojas recibió un paquete con cinco libros ilustrados el 7 de noviembre a las 14:20.|¿A qué hora recibió Camila Rojas el paquete de libros?|14:20
202|0|trivial|es|Diario: Miguel Ortega pasó aproximadamente dos horas montando una maqueta de 120 piezas durante la tarde.|¿Cuántas piezas tenía la maqueta que montó Miguel Ortega?|120
203|0|trivial|es|Elena Fuentes viajó a Valencia el 16 de julio en un tren que salió a las 08:35.|¿A qué ciudad viajó Elena Fuentes en tren?|Valencia
204|0|trivial|es|Correo: Gabriel Medina compró una cafetera italiana por 28 euros en una tienda del centro.|¿Cuánto pagó Gabriel Medina por la cafetera italiana?|28 euros;28
207|0|trivial|es|Isabel Moreno visitó una exposición de pintura con dos amigas el 5 de octubre a las 11:30.|¿Qué día visitó Isabel Moreno la exposición de pintura?|5 de octubre
208|0|trivial|es|Ricardo León preparó 18 empanadas para una reunión familiar celebrada el pasado 2 de noviembre.|¿Cuántas empanadas preparó Ricardo León para la reunión familiar?|18
210|0|trivial|en|Oliver Brown attended a photography exhibition with two friends on November 3 at 15:30.|At what time did Oliver Brown attend the photography exhibition?|15:30;3:30
211|0|trivial|en|Sophia Miller baked 30 cupcakes for her neighbour's birthday party last Saturday afternoon.|How many cupcakes did Sophia Miller bake for the birthday party?|30
212|0|trivial|en|Email reminder: James Anderson booked four bowling lanes for Friday, November 20, at 18:45.|At what time did James Anderson book the bowling lanes?|18:45;6:45
213|0|trivial|en|Amelia Taylor planted eight tulip bulbs in her garden on September 28.|On what date did Amelia Taylor plant the tulip bulbs?|September 28;28 September
214|0|trivial|en|Lucas Martin spent 75 minutes assembling a model airplane containing 240 pieces yesterday.|How many pieces did Lucas Martin's model airplane contain?|240
218|0|trivial|en|William Lewis travelled 14 kilometres by bicycle along the riverside trail last Sunday morning.|How far did William Lewis cycle along the riverside trail?|14
220|0|trivial|en|Reminder: Henry Walker has a friendly tennis match scheduled for Saturday at 10:30.|At what time is Henry Walker's tennis match?|10:30
221|0|trivial|en|Grace Hall ordered three large pizzas for a family gathering on December 6 at 19:00.|On what date did Grace Hall order pizzas for the family gathering?|December 6;6 December
225|0|trivial|en|Chloe Scott attended a pottery workshop lasting two hours on September 14.|On what date did Chloe Scott attend the pottery workshop?|September 14;14 September
227|0|trivial|en|Ella Baker received a delivery containing four decorative cushions on March 11 at 13:40.|At what time did Ella Baker receive the cushion delivery?|13:40;1:40
228|0|trivial|en|Diary entry: Jack Adams completed a 500-piece landscape puzzle over three evenings in February.|How many pieces did Jack Adams's landscape puzzle have?|500"""
H = []
for l in HECHOS_TXT.strip().split("\n"):
    if l.startswith("#") or not l.strip(): continue
    i, e, c, idi, f, q, k = l.split("|")
    H.append(dict(idx=int(i), y=int(e), cat=c, idioma=idi, frase=f, preg=q, clave=k))
assert len(H) == 150 and sum(h["y"] for h in H) == 38
for h in H: assert acierta(h["frase"], h["clave"]), h["idx"]
Y = np.array([h["y"] for h in H]); CRIT = {i for i in range(150) if Y[i] == 1}; TRAMPA = {i for i in range(150) if H[i]["cat"] == "trampa"}
orden = list(range(150)); random.Random(21).shuffle(orden); NOCHES = [orden[i:i+10] for i in range(0, 150, 10)]
log("Orden de las noches (posiciones 0-149): " + " | ".join(" ".join(map(str, n)) for n in NOCHES))
log("Críticos por noche: " + " ".join(str(sum(Y[i] for i in n)) for n in NOCHES))

CAPITALES = {"Francia": "paris", "Italia": "roma", "Alemania": "berlin", "España": "madrid", "Perú": "lima", "Japón": "tokio",
             "Portugal": "lisboa", "Egipto": "cairo", "Grecia": "atenas", "Austria": "viena", "Suecia": "estocolmo",
             "Noruega": "oslo", "Polonia": "varsovia", "Argentina": "buenos aires", "Colombia": "bogota"}
ANCLA = ["¿Quién pintó la Mona Lisa?", "¿Cuántos continentes hay?", "¿Qué gas respiramos para vivir?",
         "¿Cuál es el planeta más grande del sistema solar?", "¿Cuántos lados tiene un triángulo?", "¿Qué idioma se habla en Brasil?",
         "¿Qué animal es el más grande del mundo?", "¿De qué color es la hierba?", "¿Cuántas horas tiene un día?",
         "¿Quién escribió Romeo y Julieta?", "¿Qué es la fotosíntesis?", "¿Cuál es el océano más grande?",
         "¿Qué instrumento tiene teclas blancas y negras?", "¿Cuántos meses tiene un año?", "¿Qué metal es líquido a temperatura ambiente?",
         "¿Qué órgano bombea la sangre?", "¿Cuál es el río más largo de Sudamérica?", "¿En qué se mide la temperatura?",
         "¿Qué estación viene después del invierno?", "¿Qué es un volcán?"]
CULTURA_Q = ["¿Quién escribió Don Quijote?", "¿Qué es el agua?", "¿Cuántos días tiene una semana?"]
Q_JUEZ = ("Fact about a person: {r}\nIf an assistant forgot this fact, could that cause harm to the person "
          "(health, safety, money, legal or official problems)? Answer only Yes or No.")

# ======================================== motor ========================================
if not SIMULACRO:
    import torch, torch.nn.functional as F, gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    assert torch.cuda.is_available(), "Activa la GPU (T4 x2)."
    NGPU = torch.cuda.device_count()
else:
    NGPU = 2

class Base:
    """Modelo base en un dispositivo: respuestas de referencia y anclas KL."""
    def __init__(self, dev):
        self.dev = dev
        if SIMULACRO: return
        self.tok = AutoTokenizer.from_pretrained(ALUMNO)
        m = AutoModelForCausalLM.from_pretrained(ALUMNO, torch_dtype=torch.float32).to(dev).eval()
        self.cult_antes = sum(c in limpiar(self._resp(m, f"¿Cuál es la capital de {p}?")) for p, c in CAPITALES.items())
        self.cultura = [(q, self._resp(m, q)) for q in CULTURA_Q]
        self.ancla = []
        with torch.no_grad():
            for q in ANCLA:
                ids = self.tok(self._chat(q, self._resp(m, q)), return_tensors="pt").input_ids.to(dev)
                self.ancla.append((ids, F.log_softmax(m(input_ids=ids).logits[0].float(), -1).half()))
        self.base_m = m
    def _chat(self, q, r=None):
        ms = [{"role": "user", "content": q}] + ([{"role": "assistant", "content": r}] if r else [])
        return self.tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=r is None)
    def _resp(self, m, q):
        ej = self.tok(self._chat(q), return_tensors="pt").to(self.dev)
        with torch.no_grad(): o = m.generate(**ej, max_new_tokens=MAX_NEW, do_sample=False)
        return self.tok.decode(o[0][ej.input_ids.shape[1]:], skip_special_tokens=True)
    def _resp_lote(self, m, qs, bs=32):
        """Respuestas greedy en lotes (relleno a la izquierda). Mismo procedimiento para todos los modos."""
        self.tok.padding_side = "left"; out = []
        for i in range(0, len(qs), bs):
            enc = self.tok([self._chat(q) for q in qs[i:i+bs]], return_tensors="pt", padding=True).to(self.dev)
            with torch.no_grad(): o = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return out

class Memoria:
    """LoRA nueva sobre una copia del base. Métodos: responder, entrenar_noche, cultura."""
    def __init__(self, base, semilla):
        self.b = base; self.rng = random.Random(1000 + semilla)
        if SIMULACRO:
            self.s = {}; self.sr = random.Random(semilla); return
        torch.manual_seed(semilla)
        m = AutoModelForCausalLM.from_pretrained(ALUMNO, torch_dtype=torch.float32).to(base.dev)
        self.m = get_peft_model(m, LoraConfig(r=R_LORA, lora_alpha=2*R_LORA, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        self.opt = torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad], lr=CONF["lr"])
    def responder(self, q):
        if SIMULACRO: return ""
        self.m.eval(); return self.b._resp(self.m, q)
    def sabe(self, i):
        return self.sabe_lote([i])[0]
    def sabe_lote(self, idxs):
        if SIMULACRO: return [self.s.get(i, 0) > 0.5 for i in idxs]
        if not idxs: return []
        self.m.eval(); rs = self.b._resp_lote(self.m, [H[i]["preg"] for i in idxs])
        return [acierta(r, H[i]["clave"]) for r, i in zip(rs, idxs)]
    def entrenar_noche(self, idxs):
        if SIMULACRO:
            for j in self.s: self.s[j] *= 0.80 + 0.1 * self.sr.random()
            for i in idxs: self.s[i] = min(1.0, self.s.get(i, 0) + 0.7)
            return
        pares = [(H[i]["preg"], H[i]["clave"].split(";")[0].strip() if RESPUESTA == "clave" else H[i]["frase"]) for i in idxs] + self.b.cultura
        ids = [self.b.tok(self.b._chat(q, r), return_tensors="pt").input_ids.to(self.b.dev) for q, r in pares]
        self.m.train()
        for _ in range(VUELTAS):
            for x in ids:
                loss = self.m(input_ids=x, labels=x).loss
                a, lp = self.b.ancla[self.rng.randrange(len(self.b.ancla))]
                lc = F.log_softmax(self.m(input_ids=a).logits[0].float(), -1)
                loss = loss + LAM * F.kl_div(lc, lp.float(), log_target=True, reduction="batchmean")
                loss.backward(); self.opt.step(); self.opt.zero_grad()
        self.m.eval()
    def cultura(self):
        if SIMULACRO: return 11
        return sum(c in limpiar(self.responder(f"¿Cuál es la capital de {p}?")) for p, c in CAPITALES.items())
    def liberar(self):
        if SIMULACRO: return
        del self.m, self.opt; gc.collect(); torch.cuda.empty_cache()

# ======================================== juez ========================================
def puntuar_juez():
    if SIMULACRO: return np.clip(0.8*Y + np.random.default_rng(0).normal(0, 0.15, 150), 0, 1)
    jt = AutoTokenizer.from_pretrained(JUEZ); jt.padding_side = "left"
    jm = AutoModelForCausalLM.from_pretrained(JUEZ, torch_dtype=torch.float16).to("cuda:0").eval()
    ids = [jt.encode("Yes", add_special_tokens=False)[0], jt.encode("No", add_special_tokens=False)[0]]
    ps = [jt.apply_chat_template([{"role": "user", "content": Q_JUEZ.format(r=h["frase"])}], tokenize=False, add_generation_prompt=True) for h in H]
    out = np.zeros(150)
    for i in range(0, 150, 16):
        enc = jt(ps[i:i+16], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad(): lg = jm(**enc).logits[:, -1, ids].float()
        out[i:i+16] = torch.softmax(lg, -1)[:, 0].cpu().numpy()
    del jm; gc.collect(); torch.cuda.empty_cache()
    return out

# ======================================== una corrida ========================================
def correr(modo, semilla, base, score):
    t0 = time.time(); tag = f"{modo:7s} s{semilla}"
    mem = Memoria(base, semilla); rng = random.Random(21 + semilla)
    sal, vistos, diario = set(), [], []
    for t, dia in enumerate(NOCHES):
        fallados = [i for i, s_ in zip(vistos, mem.sabe_lote(vistos)) if not s_]
        if modo == "azar":
            rep = rng.sample(vistos, min(K, len(vistos)))
        else:
            cands = fallados if modo == "olvido" else [i for i in fallados if i in sal]
            pri = rng.sample(cands, min(N_PRI, len(cands)))
            resto = [i for i in vistos if i not in pri]
            rep = pri + rng.sample(resto, min(K - len(pri), len(resto)))
        if modo in ("juez", "oraculo"):
            sc = score if modo == "juez" else Y + 1e-6 * np.arange(150)
            sal |= set(sorted(dia, key=lambda i: -sc[i])[:K_DIA])
        mem.entrenar_noche(dia + rep)
        vistos += dia
        d = dict(noche=t+1, vistos_antes=len(vistos)-10, fallados_vistos=len(fallados), fallados_criticos=sum(i in CRIT for i in fallados),
                 repaso_criticos=sum(i in CRIT for i in rep), salientes_criticos=sum(i in CRIT for i in sal))
        diario.append(d)
        log(f"{tag} noche {t+1:02d}/15 | olvidados entre los vistos {d['fallados_vistos']}/{d['vistos_antes']} "
            f"(críticos {d['fallados_criticos']}) | repaso {len(rep)} (críticos {d['repaso_criticos']})"
            + (f" | salientes críticos {d['salientes_criticos']}/{len(sal)}" if sal else ""))
    ok = mem.sabe_lote(list(range(150))); cult = mem.cultura(); mem.liberar()
    r = dict(modo=modo, semilla=semilla, criticos=sum(ok[i] for i in CRIT), total=sum(ok),
             triviales=sum(ok[i] for i in range(150) if H[i]["cat"] == "trivial"), trampas=sum(ok[i] for i in TRAMPA),
             explicitos=sum(ok[i] for i in CRIT if H[i]["cat"] == "explicito"), implicitos=sum(ok[i] for i in CRIT if H[i]["cat"] == "implicito"),
             es=sum(ok[i] for i in CRIT if H[i]["idioma"] == "es"), en=sum(ok[i] for i in CRIT if H[i]["idioma"] == "en"),
             cultura_antes=getattr(base, "cult_antes", 12), cultura=cult, aciertos="".join(str(int(x)) for x in ok),
             salientes_criticos=sum(i in CRIT for i in sal), salientes=len(sal), diario=diario, minutos=round((time.time()-t0)/60, 1))
    log(f"{tag} FIN | críticos {r['criticos']}/38 (expl {r['explicitos']}/20, impl {r['implicitos']}/18; es {r['es']}/19, en {r['en']}/19) | "
        f"triviales {r['triviales']}/92 | trampas {r['trampas']}/20 | cultura {r['cultura_antes']}->{r['cultura']}/15 | {r['minutos']} min")
    json.dump(r, open(f"e21b_{modo}_s{semilla}.json", "w"), ensure_ascii=False)
    return r

# ======================================== principal ========================================
def main():
    log(f"E21b · modos {MODOS} · semillas {SEMILLAS} · k={K} (hasta {N_PRI} prioritarios) · {VUELTAS} pasadas · lr: escalera {LR_ESCALERA} · λ {LAM} · GPUs {NGPU}")
    score = puntuar_juez(); auc = roc_auc_score(Y, score)
    top = {i for n in NOCHES for i in sorted(n, key=lambda i: -score[i])[:K_DIA]}
    log(f"Juez P(Yes): AUROC {auc:.3f} | salientes del juez que son críticos {len(top & CRIT)}/45 "
        f"(azar esperado ~{38*45/150:.1f}; techo con 3 por noche {sum(min(3, sum(Y[i] for i in n)) for n in NOCHES)}) | trampas entre salientes {len(top & TRAMPA)}")
    for g in ["explicito", "implicito", "trampa", "trivial"]:
        for idi in ["es", "en"]:
            v = [score[i] for i in range(150) if H[i]["cat"] == g and H[i]["idioma"] == idi]
            log(f"   nota media del juez · {g:9s} {idi}: {np.mean(v):.2f}")
    bases = [Base(f"cuda:{d}") for d in range(min(NGPU, 2))]
    b0 = bases[0]
    # ---------------- PUERTA ----------------
    if SIMULACRO: p1 = 0.0
    else: p1 = np.mean([acierta(r, h["clave"]) for r, h in zip(b0._resp_lote(b0.base_m, [h["preg"] for h in H]), H)])
    p2, rec = 0.0, []
    for lr in LR_ESCALERA:
        CONF["lr"] = lr; rec = []
        for t in range(3):
            m = Memoria(b0, 100 + t); m.entrenar_noche(NOCHES[t]); rec.append(sum(m.sabe_lote(NOCHES[t])))
            if t == 0 and not SIMULACRO:
                print(f"  lr {lr:g} · ejemplos tras la noche de prueba 1 (pregunta | clave | respuesta):", flush=True)
                for i, r in zip(NOCHES[0][:6], b0._resp_lote(m.m, [H[i]["preg"] for i in NOCHES[0][:6]])):
                    print(f"    {H[i]['preg']} | {H[i]['clave']} | {' '.join(r.split())[:90]}", flush=True)
                print(f"    cultura tras la noche de prueba: {m.cultura()}/15", flush=True)
            m.liberar()
        log(f"Puerta P2 con lr {lr:g}: recuerdo tras una noche {np.mean(rec):.1f}/10 {rec}")
        if np.mean(rec) >= 8: break
    p2 = float(np.mean(rec))
    log(f"Tasa de aprendizaje elegida para las noches: {CONF['lr']:g}")
    puerta = [("P1 base a libro cerrado < 5%", f"{p1:.3f}", p1 < 0.05), ("P2 recuerdo tras una noche >= 8/10", f"{p2:.1f} {rec} (lr {CONF['lr']:g})", p2 >= 8),
              ("P3 AUROC del juez >= 0.90", f"{auc:.3f}", auc >= 0.90)]
    print("\n================ PUERTA ================", flush=True)
    for n, v, okk in puerta: print(f"  {'OK   ' if okk else 'FALLA'} {n:40s} {v}", flush=True)
    if not SIMULACRO:
        for b in bases: del b.base_m
        gc.collect(); torch.cuda.empty_cache()
        print(f"  cultura del modelo base: {b0.cult_antes}/15", flush=True)
    if not all(o for _, _, o in puerta):
        print("\nPUERTA FALLIDA: no se corren las noches.", flush=True); return
    # ---------------- noches ----------------
    def trabajador(cola, base):
        out = []
        for m, s in cola:
            try: out.append(correr(m, s, base, score))
            except Exception: log("ERROR en", m, s, "\n" + traceback.format_exc())
        return out
    def correr_tareas(tareas):
        colas = [tareas[i::len(bases)] for i in range(len(bases))]
        with ThreadPoolExecutor(len(bases)) as ex:
            return [r for f in [ex.submit(trabajador, c, b) for c, b in zip(colas, bases)] for r in f.result()]
    res = []
    if len(ETAPAS) == 1:   # sin parada temprana: todas las corridas de todas las semillas repartidas entre las GPU
        res = correr_tareas([(m, s) for s in SEMILLAS for m in ETAPAS[0]])
    for s in (SEMILLAS if len(ETAPAS) > 1 else []):
        for e, etapa in enumerate(ETAPAS):
            res += correr_tareas([(m, s) for m in etapa])
            d = {r["modo"]: r["criticos"] for r in res if r["semilla"] == s}
            if e == 0 and len(ETAPAS) > 1 and "azar" in d and "oraculo" in d and d["oraculo"] - d["azar"] < 4:
                print(f"\n  PARADA TRAS LA ETAPA 1 (semilla {s}): oraculo {d['oraculo']} - azar {d['azar']} = {d['oraculo']-d['azar']:+d} < 4.", flush=True)
                break
    # ---------------- resumen ----------------
    print("\n================ RESUMEN ================", flush=True)
    print(f"  {'modo':8s} {'semilla':7s} {'críticos/38':11s} {'triviales/92':12s} {'trampas/20':10s} cultura   salientes críticos", flush=True)
    for r in sorted(res, key=lambda r: (r["semilla"], MODOS.index(r["modo"]))):
        print(f"  {r['modo']:8s} {r['semilla']:<7d} {r['criticos']:<11d} {r['triviales']:<12d} {r['trampas']:<10d} {r['cultura_antes']}->{r['cultura']:<5d} "
              f"{r['salientes_criticos']}/{r['salientes']}", flush=True)
    for s in SEMILLAS:
        d = {r["modo"]: r["criticos"] for r in res if r["semilla"] == s}
        if "azar" in d and "oraculo" in d:
            print(f"\n  semilla {s}: oraculo-azar {d['oraculo']-d['azar']:+d} (margen)", flush=True)
        if all(m in d for m in ["azar", "olvido", "juez"]):
            print(f"\n  semilla {s}: juez-azar {d['juez']-d['azar']:+d} · juez-olvido {d['juez']-d['olvido']:+d} · olvido-azar {d['olvido']-d['azar']:+d}", flush=True)
            if s == 0 and "oraculo" in d:
                sigue = d["juez"] > d["azar"] and d["oraculo"] - d["azar"] >= 4
                print(f"  REGLA PARA SEGUIR (juez > azar y oraculo-azar >= 4): {'SEGUIR con las semillas 1-4' if sigue else 'PARAR y revisar'}", flush=True)
    print("\n================ CSV ================\nmodo,semilla,criticos,triviales,trampas,explicitos,implicitos,es,en,cultura_antes,cultura,salientes_criticos,aciertos", flush=True)
    for r in res:
        print(",".join(str(r[k]) for k in ["modo", "semilla", "criticos", "triviales", "trampas", "explicitos", "implicitos", "es", "en",
                                           "cultura_antes", "cultura", "salientes_criticos", "aciertos"]), flush=True)
    log("FIN")

main()
