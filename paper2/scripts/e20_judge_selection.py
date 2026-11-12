# -*- coding: utf-8 -*-
# =====================================================================================
# E20 · AMÍGDALA SIN ETIQUETAS HUMANAS (prueba corta) · ¿Soñar aporta algo frente a preguntar directamente?
#
# Kaggle: GPU T4 x2, Internet On. Celda 1:  !pip uninstall -y -q torchao
#         Celda 2: este archivo.  Save Version -> Save & Run All.  Unos 30-40 min. No entrena memoria.
#
# Dos maneras de puntuar la criticidad de un dato SIN ninguna etiqueta humana, con el mismo juez (Qwen2.5-1.5B)
# y los mismos cuatro valores en el prompt (salud, seguridad, dinero, problemas legales):
#   SUEÑO   : el juez imagina 8 preguntas futuras sin ver ningun dato; el 0.5B responde cada una CON y SIN el dato;
#             el juez estima el daño evitado y su gravedad.  (17 llamadas al juez + 8 respuestas por dato)
#   DIRECTO : una sola pregunta al juez: "si un asistente olvidara este dato, ¿podria causar daño?"
# Se evalua en Kalandor-100 (plantillas) y en el examen fuera de plantilla de 229 frases del paper 1
# (ese examen ya se uso una vez; aqui se reutiliza para metodos nuevos y se declara).
#
# ---------------- TODO LO SIGUIENTE QUEDA FIJADO ANTES DE LANZAR ----------------
# Comparacion primaria: AUROC(sueño) - AUROC(directo 1.5B) en el examen fuera de plantilla, bootstrap pareado.
# PUERTA: si el juez directo 1.5B no llega a 0.60 en Kalandor, el juez no sirve y el resultado es INCONCLUSO.
# VEREDICTO:
#   SUEÑO GANA   : diferencia >= +0.05 con IC95 sin el 0 en el examen, y no peor que -0.03 en Kalandor
#   DIRECTO GANA : diferencia <= -0.05 con IC95 sin el 0 en el examen
#   EMPATE       : |diferencia| < 0.03 en el examen
#   INTERMEDIO   : lo demas
# Regla para el paso siguiente (noches): el maestro sera el ganador; con empate, el directo (cuesta 17 veces menos).
# PREDICCION ESCRITA (Claude): DIRECTO GANA o EMPATE.
# Se reporta aparte: 7B directo, 0.5B directo, e5 sin entrenar, y los alumnos baratos (sonda capa 16 y e5)
# destilados de cada maestro, frente a la sonda entrenada con etiquetas humanas.
# Limites: los cuatro valores siguen escritos a mano en el prompt; las preguntas soñadas no ven a la persona.
# =====================================================================================
SIMULACRO = False
USAR_7B = True
N_ESC = 8

import random, unicodedata, time, re
import numpy as np
from sklearn.metrics import roc_auc_score
T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:5.1f} min]", *a, flush=True)
def limpiar(t):
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")

# ------------------------------- KALANDOR-100 -----------------------------------
N = ["Arlo", "Bexa", "Ciro", "Dalia", "Emre", "Fiona", "Gael", "Hana", "Iven", "Jorun"]
FEM = {"Bexa", "Dalia", "Fiona", "Hana"}
datos = []   # (pregunta, respuesta, clave, importante, grupo)

alerg = [("a la penicilina", "penicilina"), ("a las nueces", "nueces"), ("al látex", "latex"),
         ("a las picaduras de avispa", "avispa"), ("al marisco", "marisco"), ("a la aspirina", "aspirina"),
         ("al polen de abedul", "abedul"), ("a los huevos", "huevos"), ("a la amoxicilina", "amoxicilina"),
         ("al sésamo", "sesamo")]
for nm, (v, c) in zip(N, alerg):
    a = "alérgica" if nm in FEM else "alérgico"
    datos.append((f"¿A qué es {a} {nm}?", f"{nm} es {a} {v}.", c, True, "alergia"))

implicito = [
    ("¿Qué le pasa a Arlo si conduce más de una hora?", "Arlo se queda dormido si conduce más de una hora.", "dormido"),
    ("¿Qué hace la hija pequeña de Bexa con los enchufes?", "La hija pequeña de Bexa mete los dedos en los enchufes.", "dedos"),
    ("¿Cómo está la barandilla del balcón de Ciro?", "La barandilla del balcón de Ciro está suelta.", "suelta"),
    ("¿Qué le pasa a Dalia en las alturas?", "Dalia se marea y pierde el equilibrio en las alturas.", "equilibrio"),
    ("¿Qué pasa con la estufa de Emre?", "La estufa de Emre echa un olor raro cuando se enciende.", "olor raro"),
    ("¿Qué no puede comer el perro de Fiona?", "El perro de Fiona se pone muy malo si come uvas.", "uvas"),
    ("¿Cómo está el hielo del estanque donde patina Gael?", "El hielo del estanque donde patina Gael está muy fino.", "fino"),
    ("¿Qué le pasa a Hana cuando hace mucho calor?", "Hana se desmaya cuando hace mucho calor.", "desmaya"),
    ("¿Qué hace el abuelo de Iven con los fogones?", "El abuelo de Iven deja los fogones encendidos y se olvida.", "fogones"),
    ("¿Qué le pasa al bebé de Jorun con los trozos pequeños de comida?", "El bebé de Jorun se atraganta con facilidad con trozos pequeños.", "atraganta"),
]
for q, r, c in implicito:
    datos.append((q, r, c, True, "implícito"))

plazos = [
    ("¿Cuándo tiene Arlo el juicio?", "Arlo tiene un juicio el 12 de diciembre.", "12 de diciembre"),
    ("¿Cuándo vence la multa de Bexa?", "La multa de tráfico de Bexa vence el viernes y después se duplica.", "viernes"),
    ("¿Cuándo caduca el permiso de residencia de Ciro?", "El permiso de residencia de Ciro caduca en febrero.", "febrero"),
    ("¿Cuál es el PIN de la tarjeta de Dalia?", "El PIN de la tarjeta de Dalia es 5821.", "5821"),
    ("¿Cuánto le debe Emre al casero?", "Emre le debe dos meses de alquiler al casero.", "dos meses"),
    ("¿Dónde está el testamento de Fiona?", "El testamento de Fiona está en la notaría de la plaza.", "notaria"),
    ("¿Cuándo tiene que presentarse Gael en el juzgado?", "Gael tiene que presentarse en el juzgado el lunes a las nueve.", "lunes"),
    ("¿Cuándo vence el seguro de salud de Hana?", "El seguro de salud de Hana vence el 31 de enero.", "31 de enero"),
    ("¿Qué firmó Iven con el banco?", "Iven firmó como avalista del préstamo de su hermano.", "avalista"),
    ("¿Dónde tiene Jorun una citación?", "Jorun tiene una citación en la comisaría el día 3.", "comisaria"),
]
for q, r, c in plazos:
    datos.append((q, r, c, True, "plazo"))

mascotas = [("un gato llamado", "Bimo"), ("un loro llamado", "Kiko"), ("un conejo llamado", "Pelusa"),
            ("un hámster llamado", "Tobi"), ("una tortuga llamada", "Lenta"), ("un perro llamado", "Rufo"),
            ("un hurón llamado", "Chispa"), ("un canario llamado", "Piolo"), ("una cobaya llamada", "Nube"),
            ("un pez llamado", "Burbuja")]
for nm, (a, v) in zip(N, mascotas):
    datos.append((f"¿Cómo se llama la mascota de {nm}?", f"La mascota de {nm} es {a} {v}.", limpiar(v), False, "mascota"))

comidas = [("la lasaña", "lasana"), ("el sushi", "sushi"), ("las empanadas", "empanadas"), ("el ceviche", "ceviche"),
           ("la paella", "paella"), ("el ramen", "ramen"), ("los tacos", "tacos"), ("el risotto", "risotto"),
           ("el goulash", "goulash"), ("la moussaka", "moussaka")]
for nm, (v, c) in zip(N, comidas):
    datos.append((f"¿Cuál es la comida favorita de {nm}?", f"La comida favorita de {nm} es {v}.", c, False, "comida"))

colores = [("turquesa", "turquesa"), ("granate", "granate"), ("ocre", "ocre"), ("lila", "lila"), ("verde oliva", "oliva"),
           ("coral", "coral"), ("añil", "anil"), ("magenta", "magenta"), ("salmón", "salmon"), ("beige", "beige")]
for nm, (v, c) in zip(N, colores):
    datos.append((f"¿Cuál es el color favorito de {nm}?", f"El color favorito de {nm} es el {v}.", c, False, "color"))

ocio = [("hace cerámica", "ceramica"), ("va al rocódromo", "rocodromo"), ("juega al pádel", "padel"),
        ("pinta acuarelas", "acuarelas"), ("sale en kayak", "kayak"), ("hace origami", "origami"),
        ("observa pájaros", "pajaros"), ("cuida sus bonsáis", "bonsais"), ("monta puzzles", "puzzles"),
        ("hace pan casero", "pan casero")]
for nm, (v, c) in zip(N, ocio):
    datos.append((f"¿Qué hace {nm} los fines de semana?", f"Los fines de semana, {nm} {v}.", c, False, "ocio"))

instr = [("el violonchelo", "violonchelo"), ("el ukelele", "ukelele"), ("el acordeón", "acordeon"),
         ("el clarinete", "clarinete"), ("la batería", "bateria"), ("el arpa", "arpa"), ("el saxofón", "saxofon"),
         ("la flauta travesera", "travesera"), ("el banjo", "banjo"), ("la trompeta", "trompeta")]
for nm, (v, c) in zip(N, instr):
    datos.append((f"¿Qué instrumento toca {nm}?", f"{nm} toca {v}.", c, False, "instrumento"))

ciudades = ["Kioto", "Marrakech", "Cuzco", "Valparaíso", "Sevilla", "Edimburgo", "Cartagena", "Bergen", "Salzburgo", "Granada"]
for nm, v in zip(N, ciudades):
    datos.append((f"¿Qué ciudad sueña con visitar {nm}?", f"{nm} sueña con visitar {v}.", limpiar(v), False, "ciudad"))

trampas = [
    ("¿Qué colecciona Arlo?", "Arlo colecciona billetes antiguos de otros países.", "billetes"),
    ("¿Qué serie ve Bexa?", "Bexa ve una serie sobre un hospital.", "hospital"),
    ("¿Qué guarda Ciro en su hucha?", "Ciro guarda monedas de un céntimo en una hucha de barro.", "centimo"),
    ("¿Qué guarda Dalia como recuerdo?", "Dalia guarda las tarjetas de embarque de su primer viaje como recuerdo.", "tarjetas de embarque"),
    ("¿De qué trata el libro que lee Emre?", "Emre lee un libro sobre la historia de los bancos.", "historia de los bancos"),
    ("¿A qué jugaba Fiona de pequeña?", "De pequeña, Fiona jugaba a ser doctora con sus muñecas.", "doctora"),
    ("¿Qué logo tiene la camiseta de Gael?", "Gael tiene una camiseta con el logo de una farmacia.", "farmacia"),
    ("¿A qué juego de mesa le gusta jugar a Hana?", "A Hana le gusta jugar a un juego de mesa sobre la bolsa.", "bolsa"),
    ("¿Qué colecciona Iven?", "Iven colecciona recibos antiguos de tiendas que ya cerraron.", "recibos"),
    ("¿Qué dibuja Jorun en sus cuadernos?", "Jorun dibuja ambulancias en sus cuadernos.", "ambulancias"),
]
for q, r, c in trampas:
    datos.append((q, r, c, False, "trampa"))

assert len(datos) == 100
for q, r, c, imp_, g in datos:                                   # cada clave aparece en su propia respuesta
    assert c in limpiar(r), (c, r)
IMPORTANTES = {i for i, d in enumerate(datos) if d[3]}
POPULARES   = {i for i, d in enumerate(datos) if d[4] in ("mascota", "comida")}
IMPLICITOS  = {i for i, d in enumerate(datos) if d[4] == "implícito"}
TRAMPAS     = {i for i, d in enumerate(datos) if d[4] == "trampa"}

# ------------------- frases de plantilla (para destilar; ENT_AUTO va SIN etiquetas) -------------------
nombres_j = ["Garren", "Luvra", "Pim", "Odra", "Fenk", "Ysel", "Tamo", "Brin", "Kesh", "Nali", "Corvo", "Dena"]
esen_v1 = ["{n} es alérgico al marisco.", "{n} es alérgica a las abejas.", "{n} toma insulina cada noche.",
    "El PIN de la tarjeta de {n} es 3391.", "La clave del correo de {n} está en la libreta roja.",
    "El contrato de alquiler de {n} vence el 1 de junio.", "{n} tiene que pagar sus impuestos antes del 30 de abril.",
    "Si {n} tiene convulsiones, hay que ponerlo de lado y llamar al 144.", "{n} no puede tomar ibuprofeno.",
    "El número de póliza del seguro de {n} es 77-120.", "El DNI de {n} caduca en enero.",
    "El médico le prohibió a {n} conducir de noche."]
triv_v1 = ["A {n} le gusta el color naranja.", "{n} colecciona postales.", "{n} desayuna avena.",
    "La película favorita de {n} es de piratas.", "{n} pasea por el parque los domingos.",
    "{n} tiene una planta llamada Rulo.", "{n} prefiere el invierno al verano.", "A {n} le gusta silbar.",
    "{n} tiene una taza amarilla.", "{n} escucha jazz al cocinar.", "{n} juega a las cartas los viernes.",
    "El sofá de {n} es gris."]
esen_v2 = ["{n} no sabe nadar.", "El hijo pequeño de {n} se sube a las ventanas.",
    "{n} se marea si pasa muchas horas sin comer.", "El perro de {n} muerde a los desconocidos.",
    "Los frenos de la bici de {n} casi no funcionan.", "{n} tiene que recoger a su abuela del hospital el viernes."]
triv_v2 = ["{n} vio un documental sobre bancos suizos.", "{n} sueña con visitar Japón.",
    "{n} tiene una taza con forma de billete.", "{n} jugaba a médicos de pequeño.",
    "{n} colecciona etiquetas de maletas de hoteles.", "{n} juega al Monopoly los domingos."]
def hacer(esen, triv, reps=4, semilla=1):
    r = random.Random(semilla)
    return [(t.format(n=r.choice(nombres_j)), 1) for t in esen for _ in range(reps)] + \
           [(t.format(n=r.choice(nombres_j)), 0) for t in triv for _ in range(reps)]
ENT_JUEZ = hacer(esen_v1 + esen_v2, triv_v1 + triv_v2)
ENT_AUTO = [t for t, _ in hacer(esen_v1 + esen_v2, triv_v1 + triv_v2, reps=2, semilla=7)]   # SIN etiquetas

# ------------------- examen fuera de plantilla (229 frases, congelado en el paper 1) -------------------
EXAMEN = """
1|explicito|es|gpt|Mariana Torres es alérgica a la penicilina y debe informar de ello antes de cualquier tratamiento médico.
1|explicito|es|gpt|Javier Mendoza debe tomar dos comprimidos de anticoagulante cada mañana, según su prescripción médica.
1|explicito|es|gpt|URGENTE: Lucía Fernández tiene hasta el 18 de noviembre para presentar su recurso judicial.
1|explicito|es|gpt|La contraseña del archivo bancario de Roberto Salinas es NubeAzul927!
1|explicito|es|gpt|Claudia Ramírez tiene una cita con el juez el próximo martes a las 09:30.
1|explicito|es|gpt|Miguel Herrera debe transferir 4.800 euros antes del 25 de noviembre para evitar una penalización contractual.
1|explicito|es|gpt|En su historial médico consta que Patricia Gómez es alérgica al látex.
1|explicito|es|gpt|Nota personal: Andrés Castillo tiene pendiente pagar 1.250 euros de alquiler antes del día 15.
1|explicito|es|gpt|Elena Vargas utiliza insulina diariamente y necesita conservarla refrigerada durante sus viajes.
1|explicito|es|gpt|El plazo de apelación de Fernando Ruiz vence definitivamente el 30 de noviembre.
1|explicito|es|gpt|Sofía Navarro tiene programada una intervención quirúrgica el 21 de noviembre a las 07:00.
1|explicito|es|gpt|Correo: Daniel Ortega debe renovar su permiso de residencia antes del 12 de diciembre.
1|explicito|es|gpt|Isabel Moreno no puede consumir frutos secos debido a una alergia alimentaria grave.
1|explicito|es|gpt|Recordatorio: Pablo Serrano tiene una audiencia judicial obligatoria el 3 de diciembre.
1|explicito|es|gpt|La tarjeta de acceso de Natalia Cabrera utiliza el código confidencial 583194.
1|explicito|en|gpt|Emily Carter has a severe allergy to peanuts and carries emergency epinephrine everywhere.
1|explicito|en|gpt|Michael Bennett must take 10 mg of blood pressure medication every evening.
1|explicito|en|gpt|URGENT: Olivia Parker must submit her legal appeal before November 22.
1|explicito|en|gpt|The password for James Wilson's encrypted backup is SilverFox841!
1|explicito|en|gpt|Sophia Mitchell has a mandatory court hearing scheduled for December 4 at 10:00 a.m.
1|explicito|en|gpt|Please remember that Daniel Brooks owes $3,600 by November 30 to avoid eviction proceedings.
1|explicito|en|gpt|Medical note: Rebecca Adams is allergic to penicillin and must avoid related antibiotics.
1|explicito|en|gpt|Thomas Reed's passport expires on December 15, making renewal urgent.
1|explicito|en|gpt|Diary entry: Grace Collins received a prescription for daily thyroid medication, 75 micrograms.
1|explicito|en|gpt|Alexander Hughes must attend his immigration appointment on November 19 at 8:45 a.m.
1|explicito|en|gpt|The outstanding invoice of $7,250 must be paid by November 28.
1|explicito|en|gpt|Charlotte Evans cannot receive MRI contrast containing gadolinium due to a documented severe reaction.
1|explicito|en|gpt|Email reminder: William Foster's probation review takes place on December 9.
1|explicito|en|gpt|Ava Richardson has a scheduled chemotherapy session on November 17 at 11:30 a.m.
1|explicito|en|gpt|The confidential alarm code assigned to Benjamin Cooper is 729461.
1|implicito|es|gpt|El frigorífico de Diego Molina dejó de enfriar ayer, aunque la luz interior sigue funcionando.
1|implicito|es|gpt|Laura Benítez ha recibido tres cartas del banco con importes pendientes que no reconoce.
1|implicito|es|gpt|El hijo de Sergio Paredes encuentra siempre las pastillas de su padre sobre la mesa del desayuno.
1|implicito|es|gpt|Nuria Campos ha empezado a quedarse sin aire después de caminar unos pocos metros.
1|implicito|es|gpt|El contrato de alquiler de Alberto Fuentes termina mañana y todavía no ha encontrado otro alojamiento.
1|implicito|es|gpt|Cada vez que Ana Beltrán come en ese restaurante, aparecen ronchas en sus brazos.
1|implicito|es|gpt|Pedro Lozano lleva semanas recibiendo llamadas de personas que preguntan por préstamos que nunca solicitó.
1|implicito|es|gpt|La puerta del apartamento de Carmen Vidal no cierra bien desde que cambiaron la cerradura.
1|implicito|es|gpt|Raúl Medina olvidó dónde guardó su pasaporte y su vuelo internacional sale dentro de seis horas.
1|implicito|es|gpt|El nivel de glucosa de Teresa Rojas aparece muy por debajo de sus registros habituales esta mañana.
1|implicito|es|gpt|Marcos Aguilar ha recibido una notificación judicial que menciona una comparecencia para el día siguiente.
1|implicito|es|gpt|Desde que instalaron la estufa, los habitantes de la vivienda despiertan con dolor de cabeza.
1|implicito|es|gpt|Elena Prieto lleva cuatro meses utilizando la misma contraseña para acceder a todas sus cuentas bancarias.
1|implicito|es|gpt|La transferencia de Roberto León figura como pendiente desde hace dos semanas y el destinatario no reconoce el pago.
1|implicito|en|gpt|David Thompson has been feeling dizzy whenever he stands up after taking his usual morning tablets.
1|implicito|en|gpt|The smoke detector in Olivia Bennett's apartment has been disconnected since the kitchen renovation.
1|implicito|en|gpt|James Walker noticed that his employer deducted the same pension contribution twice from his latest paycheck.
1|implicito|en|gpt|Sophie Turner keeps receiving password-reset emails for accounts she has not accessed recently.
1|implicito|en|gpt|The temperature inside Michael Harris's medication refrigerator has remained above its usual setting overnight.
1|implicito|en|gpt|Emma Collins's landlord has scheduled a property inspection for tomorrow, although she has not received written notice.
1|implicito|en|gpt|William Scott has started coughing whenever he enters the basement where the boiler is located.
1|implicito|en|gpt|Ava Peterson's immigration application still shows an incomplete document despite yesterday's submission confirmation.
1|implicito|en|gpt|Daniel Foster's car brakes make a grinding noise whenever he approaches a traffic light.
1|implicito|en|gpt|Rachel Adams has not received her prescribed medication refill, and the remaining supply contains only one dose.
1|implicito|en|gpt|The bank statement shows several small withdrawals from Peter Morgan's account while he was abroad.
1|implicito|en|gpt|Chloe Edwards noticed that the emergency exit in her office building is blocked by stacked delivery boxes.
1|implicito|en|gpt|Benjamin Clark's court correspondence was delivered to his previous address after he moved last month.
1|implicito|en|gpt|Isabella Wright's toddler has been unusually sleepy since drinking from an unlabeled bottle in the garage.
0|trampa|es|gpt|A Natalia Romero le encanta ver documentales sobre hospitales y descubrir cómo funcionan las consultas médicas.
0|trampa|es|gpt|Carlos Méndez siempre juega a ser abogado cuando organiza partidas de juegos de mesa con sus amigos.
0|trampa|es|gpt|A Julia Herrera le interesa aprender conceptos financieros para entender mejor las noticias económicas.
0|trampa|es|gpt|Andrés Molina colecciona fotografías antiguas de edificios que anteriormente funcionaban como juzgados.
0|trampa|es|gpt|María Castillo disfruta leyendo novelas policiacas ambientadas en hospitales universitarios.
0|trampa|es|gpt|A Diego Navarro le gusta comparar las estadísticas financieras de los equipos de fútbol.
0|trampa|es|gpt|Lucía Fernández escucha podcasts sobre historias legales mientras prepara el desayuno.
0|trampa|es|gpt|Tomás Aguilar quiere estudiar medicina algún día porque disfruta aprendiendo sobre anatomía humana.
0|trampa|es|gpt|Valentina Cruz tiene una libreta donde apunta expresiones jurídicas interesantes que encuentra en películas.
0|trampa|es|gpt|A Gabriel Torres le divierte jugar al Monopoly y administrar bancos imaginarios.
0|trampa|es|gpt|Camila Rojas suele visitar museos que tienen exposiciones sobre la historia de la medicina.
0|trampa|es|gpt|Javier Ponce está leyendo un libro introductorio sobre economía y mercados financieros.
0|trampa|es|gpt|Elena Suárez tiene como pasatiempo resolver crucigramas con vocabulario relacionado con el derecho.
0|trampa|es|gpt|A Martín Vega le gustan las películas en las que los protagonistas trabajan en tribunales.
0|trampa|es|gpt|Paula Medina sigue varias cuentas de educación financiera porque disfruta aprendiendo sobre inversiones.
0|trampa|en|gpt|Olivia Brooks enjoys watching medical dramas set in large teaching hospitals.
0|trampa|en|gpt|Ethan Parker collects miniature courtroom models as a hobby.
0|trampa|en|gpt|Sophia Reed likes reading introductory books about financial markets during her holidays.
0|trampa|en|gpt|Liam Foster regularly watches documentaries about the history of legal systems.
0|trampa|en|gpt|Amelia Carter enjoys playing board games that involve managing imaginary bank accounts.
0|trampa|en|gpt|Noah Mitchell follows several doctors on social media for their educational videos.
0|trampa|en|gpt|Grace Turner is fascinated by courtroom scenes in classic television series.
0|trampa|en|gpt|Mason Collins recently bought a notebook to collect interesting medical terminology.
0|trampa|en|gpt|Isabella Wright enjoys visiting museums featuring historical financial documents.
0|trampa|en|gpt|Lucas Bennett listens to podcasts about famous legal cases while commuting.
0|trampa|en|gpt|Charlotte Evans likes solving crossword puzzles containing economics vocabulary.
0|trampa|en|gpt|Benjamin Cooper dreams of writing a novel about a fictional hospital.
0|trampa|en|gpt|Mia Richardson enjoys learning how insurance companies calculate their prices.
0|trampa|en|gpt|Alexander Hughes watches investment tutorials because he finds stock market charts interesting.
0|trampa|en|gpt|Emily Watson often plays doctor with her younger cousins during family gatherings.
0|trivial|es|gpt|A Daniela Torres le encanta tomar café con leche mientras lee novelas los domingos por la mañana.
0|trivial|es|gpt|Javier Morales colecciona imanes de nevera de todas las ciudades que visita durante sus vacaciones.
0|trivial|es|gpt|Mariana López suele escuchar música indie mientras prepara la cena después del trabajo.
0|trivial|es|gpt|A Roberto Sánchez le gusta jugar al tenis con sus amigos los sábados por la tarde.
0|trivial|es|gpt|Lucía Herrera tiene tres plantas de interior y disfruta cuidándolas cada mañana.
0|trivial|es|gpt|Fernando Castillo prefiere las películas de ciencia ficción a las comedias románticas.
0|trivial|es|gpt|Claudia Ramírez aprendió a preparar sushi durante un viaje que hizo a Japón.
0|trivial|es|gpt|A Nicolás Vargas le encanta salir a caminar por la playa cuando está de vacaciones.
0|trivial|es|gpt|Valeria Mendoza practica yoga dos veces por semana en un estudio cerca de su casa.
0|trivial|es|gpt|Pedro Fernández tiene una colección de más de cincuenta discos de vinilo.
0|trivial|es|gpt|Sofía Aguilar disfruta fotografiando edificios antiguos y calles pequeñas durante sus paseos.
0|trivial|es|gpt|Andrés Navarro suele desayunar tostadas con aguacate y zumo de naranja.
0|trivial|es|gpt|Camila Rojas es aficionada a la jardinería y cultiva tomates en su balcón.
0|trivial|es|gpt|Miguel Ortega juega videojuegos de estrategia después de terminar sus tareas domésticas.
0|trivial|es|gpt|Elena Fuentes prefiere viajar en tren porque disfruta contemplando los paisajes.
0|trivial|es|gpt|A Gabriel Medina le gusta cocinar platos italianos y probar recetas nuevas cada semana.
0|trivial|es|gpt|Paula Serrano asiste a clases de baile latino los miércoles por la noche.
0|trivial|es|gpt|Tomás Cabrera tiene un perro llamado Bruno que suele acompañarlo durante sus caminatas.
0|trivial|es|gpt|Isabel Moreno disfruta leyendo novelas históricas antes de quedarse dormida.
0|trivial|es|gpt|Diego Salazar siempre pide helado de pistacho cuando visita una heladería.
0|trivial|es|gpt|Natalia Paredes aprendió francés viendo películas con subtítulos durante su adolescencia.
0|trivial|es|gpt|Martín Reyes juega al ajedrez online durante sus descansos del trabajo.
0|trivial|es|gpt|A Carolina Vega le encanta visitar mercados locales y descubrir productos artesanales.
0|trivial|es|gpt|Ricardo León suele salir a correr por el parque antes de comenzar su jornada.
0|trivial|es|gpt|Teresa Molina disfruta organizando cenas temáticas para celebrar los cumpleaños de sus amigos.
0|trivial|en|gpt|Emma Wilson enjoys baking chocolate cookies on rainy Sunday afternoons.
0|trivial|en|gpt|Oliver Brown collects vintage postcards from different countries.
0|trivial|en|gpt|Sophia Miller usually listens to jazz while working on her paintings.
0|trivial|en|gpt|James Anderson plays basketball with his colleagues every Thursday evening.
0|trivial|en|gpt|Amelia Taylor has recently started learning how to play the piano.
0|trivial|en|gpt|Lucas Martin prefers spending his holidays in small coastal villages.
0|trivial|en|gpt|Charlotte White enjoys watching nature documentaries before going to bed.
0|trivial|en|gpt|Benjamin Harris grows fresh basil and mint in his kitchen.
0|trivial|en|gpt|Isabella Clark loves visiting independent bookstores whenever she travels.
0|trivial|en|gpt|William Lewis usually drinks green tea with breakfast.
0|trivial|en|gpt|Mia Robinson takes photographs of sunsets and shares them with her friends.
0|trivial|en|gpt|Henry Walker enjoys playing chess at the local community centre.
0|trivial|en|gpt|Grace Hall spends her weekends exploring hiking trails in the countryside.
0|trivial|en|gpt|Daniel Young has a collection of colourful football jerseys.
0|trivial|en|gpt|Ava King likes preparing homemade pasta for her family.
0|trivial|en|gpt|Samuel Wright watches comedy films whenever he needs a relaxing evening.
0|trivial|en|gpt|Chloe Scott enjoys swimming at the beach during the summer.
0|trivial|en|gpt|Matthew Green plays the guitar in a small amateur band.
0|trivial|en|gpt|Ella Baker likes decorating her apartment with second-hand furniture.
0|trivial|en|gpt|Jack Adams often visits art galleries with his older sister.
0|trivial|en|gpt|Lily Nelson enjoys making handmade greeting cards for special occasions.
0|trivial|en|gpt|David Carter prefers cycling to work whenever the weather is sunny.
0|trivial|en|gpt|Zoe Mitchell has a habit of doing crossword puzzles during breakfast.
0|trivial|en|gpt|Ryan Phillips loves trying different types of coffee from local cafés.
0|trivial|en|gpt|Hannah Evans spends her free time knitting scarves and colourful blankets.
0|trivial|en|gpt|Nathan Cooper enjoys playing board games with his cousins during family gatherings.
0|trivial|es|gpt|Adriana Campos disfruta escuchando música clásica mientras ordena su habitación los viernes.
0|trivial|es|gpt|Esteban Fuentes colecciona pequeñas figuras de animales que compra durante sus viajes.
0|trivial|es|gpt|Aitana Romero suele preparar batidos de fresa y plátano después de entrenar.
0|trivial|es|gpt|Manuel Cabrera es aficionado a los documentales sobre exploraciones espaciales.
0|trivial|es|gpt|Beatriz Salinas disfruta paseando por mercadillos de antigüedades los domingos.
0|trivial|es|gpt|Hugo Delgado aprendió a hacer malabares durante un campamento de verano.
0|trivial|es|gpt|Rocío Méndez tiene la costumbre de escribir pequeñas historias antes de dormir.
0|trivial|es|gpt|A Samuel Ibarra le encanta probar diferentes variedades de queso.
0|trivial|es|gpt|Inés Valdés practica escalada en rocódromo durante sus tardes libres.
0|trivial|es|gpt|Pablo Fuentes siempre lleva una libreta para dibujar paisajes durante sus viajes.
0|trivial|es|gpt|A Marina Soler le gustan especialmente las canciones de los años ochenta.
0|trivial|es|gpt|Eduardo Ponce prepara pan casero cada dos fines de semana.
0|trivial|es|gpt|Celia Domínguez disfruta jugando al voleibol en la playa con sus primos.
0|trivial|es|gpt|Iván Guerrero tiene una pequeña colección de cámaras fotográficas analógicas.
0|trivial|es|gpt|A Mónica Vidal le encanta visitar zoológicos y observar los pingüinos.
0|trivial|es|gpt|Alonso Prieto suele escuchar audiolibros mientras viaja en autobús.
0|trivial|es|gpt|Nerea Gil está aprendiendo a preparar cócteles sin alcohol en casa.
0|trivial|es|gpt|A Sergio Luna le gusta construir maquetas de aviones durante el invierno.
0|trivial|es|gpt|Alicia Pastor disfruta haciendo excursiones en bicicleta por pueblos pequeños.
0|trivial|es|gpt|Rubén Márquez toca el saxofón en una banda amateur.
0|trivial|es|gpt|Violeta Arias suele comprar flores frescas para decorar su salón.
0|trivial|es|gpt|A Marcos Nieto le encanta observar las estrellas desde su terraza.
0|trivial|es|gpt|Silvia Cortés dedica sus tardes libres a aprender cerámica.
0|trivial|es|gpt|Gonzalo Rivas disfruta preparando pizzas caseras con ingredientes originales.
0|trivial|es|gpt|Eva Montes tiene la costumbre de visitar una cafetería diferente cada sábado.
0|trivial|en|gpt|Liam Richardson enjoys building miniature wooden ships in his spare time.
0|trivial|en|gpt|Ruby Thompson collects colourful bookmarks from independent bookstores.
0|trivial|en|gpt|Oscar Bennett likes watching cooking competitions on television.
0|trivial|en|gpt|Freya Morgan spends her Sunday mornings practising calligraphy.
0|trivial|en|gpt|Elliot Hughes enjoys kayaking on calm lakes during summer holidays.
0|trivial|en|gpt|Lucy Harrison keeps a scrapbook filled with concert tickets and photographs.
0|trivial|en|gpt|George Phillips likes experimenting with homemade ice cream flavours.
0|trivial|en|gpt|Poppy Richardson attends pottery workshops twice a month.
0|trivial|en|gpt|Arthur Bailey enjoys reading biographies of famous explorers.
0|trivial|en|gpt|Daisy Cooper often goes roller skating with her neighbours.
0|trivial|en|gpt|Freddie Ward plays table tennis during his lunch breaks.
0|trivial|en|gpt|Alice Morgan loves visiting botanical gardens in different cities.
0|trivial|en|gpt|Theo Jenkins has recently developed an interest in birdwatching.
0|trivial|en|gpt|Millie Foster enjoys making homemade candles with scented oils.
0|trivial|en|gpt|Oscar Hughes usually spends Saturday evenings playing the drums.
0|trivial|en|gpt|Violet Brooks likes collecting recipes from international food magazines.
0|trivial|en|gpt|Archie Russell enjoys taking long walks through forest trails.
0|trivial|en|gpt|Florence Reed practises watercolor painting during her free afternoons.
0|trivial|en|gpt|Caleb Morris has a habit of solving Sudoku puzzles on train journeys.
0|trivial|en|gpt|Esme Collins enjoys attending outdoor music festivals with friends.
0|trivial|en|gpt|Jasper Wood collects unusual coffee mugs from souvenir shops.
0|trivial|en|gpt|Sienna Bailey likes learning traditional dances from different cultures.
0|trivial|en|gpt|Felix Grant spends his evenings restoring old wooden furniture.
0|trivial|en|gpt|Harriet Cole enjoys making fresh lemonade during warm afternoons.
0|trivial|en|gpt|Miles Turner regularly visits his local library to borrow travel magazines.
0|trivial|es|gpt|Nota: Daniela Torres compró una bicicleta de segunda mano por 180 euros durante el mercadillo del pasado sábado.
0|trivial|es|gpt|Javier Morales asistió a un concierto de jazz el 14 de octubre a las 20:30 con tres compañeros.
0|trivial|es|gpt|Mariana López preparó 24 magdalenas para la fiesta de cumpleaños de su hermana el domingo pasado.
0|trivial|es|gpt|Correo: Roberto Sánchez reservó una mesa para cuatro personas el 22 de noviembre a las 19:45.
0|trivial|es|gpt|Lucía Herrera plantó seis girasoles en las macetas de su balcón el pasado 3 de mayo.
0|trivial|es|gpt|Fernando Castillo terminó de leer una novela de 386 páginas durante sus vacaciones de agosto.
0|trivial|es|gpt|Claudia Ramírez visitó el museo de ciencias el 9 de septiembre junto con sus dos sobrinos.
0|trivial|es|gpt|Recordatorio: Nicolás Vargas tiene una partida de ajedrez amistosa el sábado a las 16:00 en el café central.
0|trivial|es|gpt|Valeria Mendoza compró tres entradas de cine por 36 euros para la sesión del viernes a las 21:15.
0|trivial|es|gpt|Pedro Fernández preparó una pizza casera de 32 centímetros para cenar el pasado miércoles.
0|trivial|es|gpt|Sofía Aguilar tomó 47 fotografías durante una excursión por los jardines botánicos el 12 de junio.
0|trivial|es|gpt|Andrés Navarro recorrió 8 kilómetros en bicicleta el domingo por la mañana antes del desayuno.
0|trivial|es|gpt|Camila Rojas recibió un paquete con cinco libros ilustrados el 7 de noviembre a las 14:20.
0|trivial|es|gpt|Diario: Miguel Ortega pasó aproximadamente dos horas montando una maqueta de 120 piezas durante la tarde.
0|trivial|es|gpt|Elena Fuentes viajó a Valencia el 16 de julio en un tren que salió a las 08:35.
0|trivial|es|gpt|Correo: Gabriel Medina compró una cafetera italiana por 28 euros en una tienda del centro.
0|trivial|es|gpt|Paula Serrano participó en una clase de salsa de 90 minutos el pasado jueves a las 18:00.
0|trivial|es|gpt|Tomás Cabrera llevó a su perro Bruno al parque durante 45 minutos el martes por la tarde.
0|trivial|es|gpt|Isabel Moreno visitó una exposición de pintura con dos amigas el 5 de octubre a las 11:30.
0|trivial|es|gpt|Ricardo León preparó 18 empanadas para una reunión familiar celebrada el pasado 2 de noviembre.
0|trivial|en|gpt|Emma Wilson bought a second-hand bookshelf for $65 at the weekend market on October 12.
0|trivial|en|gpt|Oliver Brown attended a photography exhibition with two friends on November 3 at 15:30.
0|trivial|en|gpt|Sophia Miller baked 30 cupcakes for her neighbour's birthday party last Saturday afternoon.
0|trivial|en|gpt|Email reminder: James Anderson booked four bowling lanes for Friday, November 20, at 18:45.
0|trivial|en|gpt|Amelia Taylor planted eight tulip bulbs in her garden on September 28.
0|trivial|en|gpt|Lucas Martin spent 75 minutes assembling a model airplane containing 240 pieces yesterday.
0|trivial|en|gpt|Charlotte White visited the local aquarium with her three cousins on August 16 at 11:00.
0|trivial|en|gpt|Diary entry: Benjamin Harris made 12 jars of homemade strawberry jam during the afternoon of June 8.
0|trivial|en|gpt|Isabella Clark purchased two notebooks and five coloured pencils for $19.50 at a stationery shop.
0|trivial|en|gpt|William Lewis travelled 14 kilometres by bicycle along the riverside trail last Sunday morning.
0|trivial|en|gpt|Mia Robinson took 63 photographs during a weekend visit to the botanical gardens.
0|trivial|en|gpt|Reminder: Henry Walker has a friendly tennis match scheduled for Saturday at 10:30.
0|trivial|en|gpt|Grace Hall ordered three large pizzas for a family gathering on December 6 at 19:00.
0|trivial|en|gpt|Daniel Young borrowed seven comic books from the public library on November 9.
0|trivial|en|gpt|Ava King travelled to Brighton on October 21 aboard a train departing at 09:15.
0|trivial|en|gpt|Samuel Wright spent $42 on a vintage record at an antique shop last Thursday.
0|trivial|en|gpt|Chloe Scott attended a pottery workshop lasting two hours on September 14.
0|trivial|en|gpt|Matthew Green prepared 16 sandwiches for a picnic organised with six university friends.
0|trivial|en|gpt|Ella Baker received a delivery containing four decorative cushions on March 11 at 13:40.
0|trivial|en|gpt|Diary entry: Jack Adams completed a 500-piece landscape puzzle over three evenings in February.
"""

EX = [l.split("|", 4) for l in EXAMEN.strip().split("\n")]; assert len(EX) == 229
SETS = {"kalandor": dict(txt=[d[1] for d in datos], y=np.array([int(d[3]) for d in datos]), cat=[d[4] for d in datos]),
        "examen":   dict(txt=[e[4] for e in EX], y=np.array([int(e[0]) for e in EX]), cat=[e[1]+"-"+e[2] for e in EX])}
ENT_H = [t for t, _ in ENT_JUEZ]; Y_H = np.array([y for _, y in ENT_JUEZ], float)
TODO = SETS["kalandor"]["txt"] + SETS["examen"]["txt"] + ENT_AUTO; nK, nE, nA = 100, 229, len(ENT_AUTO)
log(f"datos: Kalandor {nK} · examen {nE} · frases para destilar {nA} · frases con etiqueta humana (referencia) {len(ENT_H)}")
srng = np.random.default_rng(0); Y_TODO = np.concatenate([SETS["kalandor"]["y"], SETS["examen"]["y"], np.zeros(nA)])

if not SIMULACRO:
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    assert torch.cuda.is_available(), "Activa la GPU (T4 x2) en el notebook."
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16).to("cuda").eval()
    jtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    juez = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16).to("cuda").eval()
    log("modelos cargados (0.5B y juez 1.5B)")
    def plantilla(t, x): return t.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True)
    def gen(m, t, textos, nt=80, bs=32, muestreo=False):
        t.padding_side = "left"; ps = [plantilla(t, x) for x in textos]; orden = np.argsort([len(p) for p in ps]); out = [None]*len(ps)
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if muestreo else dict(do_sample=False)
        for i in range(0, len(ps), bs):
            idx = orden[i:i+bs]; enc = t([ps[k] for k in idx], return_tensors="pt", padding=True).to(m.device)
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=nt, **kw)
            for k, s in zip(idx, t.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)): out[k] = s.strip()
        return out
    def p_si(m, t, textos, bs=32):
        t.padding_side = "left"; ids = [t.encode("Yes", add_special_tokens=False)[0], t.encode("No", add_special_tokens=False)[0]]
        ps = [plantilla(t, x) for x in textos]; orden = np.argsort([len(p) for p in ps]); out = np.zeros(len(ps))
        for i in range(0, len(ps), bs):
            idx = orden[i:i+bs]; enc = t([ps[k] for k in idx], return_tensors="pt", padding=True).to(m.device)
            with torch.no_grad(): lg = m(**enc).logits[:, -1, ids].float()
            out[idx] = torch.softmax(lg, -1)[:, 0].cpu().numpy()
        return out
    def rep(textos, bs=32):
        tok.padding_side = "right"; out = []
        for i in range(0, len(textos), bs):
            enc = tok(textos[i:i+bs], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad(): h = base(**enc, output_hidden_states=True).hidden_states[16].float()
            mk = enc.attention_mask.unsqueeze(-1).float(); out.append(((h*mk).sum(1)/mk.sum(1)).cpu().numpy())
        return np.vstack(out)

# ------------------------------- el sueño -------------------------------
GENERICAS = ["¿Qué puedo cocinar para esta persona esta noche?", "Voy a organizar una excursión con esta persona, ¿qué planeo?",
             "¿Qué le regalo a esta persona por su cumpleaños?", "Esta persona me ha pedido ayuda con sus papeles esta semana, ¿qué hago?",
             "Voy a cuidar de esta persona y de su familia este fin de semana, ¿qué debo tener en cuenta?",
             "Esta persona viene de viaje conmigo, ¿qué preparo?", "Tengo que acompañar a esta persona al médico, ¿qué debo saber?",
             "Voy a ayudar a esta persona con sus pagos de este mes, ¿por dónde empiezo?"]
def preg(s, dato=None): return f"Lo que sabes sobre esta persona: {dato if dato else '(nada)'}\n\nPregunta: {s}\nResponde en 2 o 3 frases."
def q_dano(r, s, R): return (f"True fact about a person: {r}\nQuestion asked to an assistant: {s}\nAssistant answer: {R}\n"
                             "Given the true fact, could following this answer cause harm to the person (health, safety, money, legal or official problems)? Answer only Yes or No.")
def q_grav(r, s, R): return (f"True fact about a person: {r}\nQuestion: {s}\nAnswer given without knowing the fact: {R}\n"
                             "If this caused harm to the person, would it be serious or hard to undo (injury, hospital, legal penalty, large or permanent loss)? Answer only Yes or No.")
def q_dir(r): return (f"Fact about a person: {r}\nIf an assistant forgot this fact, could that cause harm to the person "
                      "(health, safety, money, legal or official problems)? Answer only Yes or No.")
M = len(TODO)
if SIMULACRO:
    ESC = GENERICAS[:N_ESC]; ruido = lambda f: srng.normal(size=M) + f*Y_TODO
    infl, dano, grav, dmed = ruido(.3), np.clip(ruido(1.2), 0, None), np.clip(ruido(.8), 0, 1), ruido(1.0)
    dir15, dir05, dir7 = ruido(1.6), ruido(.2), ruido(2.5); e5cos = ruido(1.0)
    XL = srng.normal(size=(M, 32)) + np.outer(Y_TODO, np.ones(32))*.5; XE = srng.normal(size=(M, 32)) + np.outer(Y_TODO, np.ones(32))*.5
    XLh = srng.normal(size=(len(ENT_H), 32)) + np.outer(Y_H, np.ones(32))*.5; XEh = XLh.copy()
    XL[nK+nE:] += np.outer(srng.integers(0, 2, nA), np.ones(32))*.5
else:
    torch.manual_seed(0)
    txt = gen(juez, jtok, ["Escribe 14 preguntas breves y distintas que alguien podría hacerle mañana a un asistente que ayuda o cuida a una "
              "persona: comidas, salidas, viajes, regalos, trámites, dinero, salud o cuidados. Refiérete a ella siempre como «esta persona». "
              "No inventes ningún dato sobre ella. Una pregunta por línea, sin numerar."], nt=400, muestreo=True)[0]
    qs = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.split("\n")]
    ESC = list(dict.fromkeys(q for q in qs if "esta persona" in q.lower() and q.endswith("?") and len(q) > 15))[:N_ESC]
    print(f"  preguntas soñadas por el juez: {len(ESC)} de {N_ESC} (el resto, de la lista fija)")
    ESC += [g for g in GENERICAS if g not in ESC][:N_ESC-len(ESC)]
    for s in ESC: print("    ·", s)
    R0 = gen(base, tok, [preg(s) for s in ESC]); log("respuestas sin dato")
    R1 = gen(base, tok, [preg(s, r) for r in TODO for s in ESC]); log("respuestas con dato")
    e0 = rep(R0); e1 = rep(R1).reshape(M, N_ESC, -1)
    cosv = (e1*e0[None]).sum(-1)/(np.linalg.norm(e1, axis=-1)*np.linalg.norm(e0, axis=-1)[None]+1e-9); infl = (1-cosv).mean(1)
    h0 = p_si(juez, jtok, [q_dano(r, s, R0[j]) for r in TODO for j, s in enumerate(ESC)]).reshape(M, N_ESC)
    h1 = p_si(juez, jtok, [q_dano(r, s, R1[i*N_ESC+j]) for i, r in enumerate(TODO) for j, s in enumerate(ESC)]).reshape(M, N_ESC)
    d = np.clip(h0-h1, 0, None); mej = d.argmax(1); dano = d.max(1); dmed = d.mean(1)
    grav = p_si(juez, jtok, [q_grav(r, ESC[mej[i]], R0[mej[i]]) for i, r in enumerate(TODO)]); log("sueño juzgado")
    dir15 = p_si(juez, jtok, [q_dir(r) for r in TODO]); dir05 = p_si(base, tok, [q_dir(r) for r in TODO]); log("juez directo")
    XL = rep(TODO); XLh = rep(ENT_H)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
    XE = st.encode(["passage: "+t for t in TODO], normalize_embeddings=True, batch_size=64); XEh = st.encode(["passage: "+t for t in ENT_H], normalize_embeddings=True, batch_size=64)
    qv = st.encode(["query: Un dato que afecta a la salud, la seguridad, el dinero o la situación legal de una persona."], normalize_embeddings=True)[0]
    e5cos = XE @ qv; del st; log("lecturas baratas")
    dir7 = None
    if USAR_7B:
        try:
            del juez; torch.cuda.empty_cache()
            t7 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            m7 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto", max_memory={0: "11GiB", 1: "14GiB"}).eval()
            dir7 = p_si(m7, t7, [q_dir(r) for r in TODO], bs=16); del m7; torch.cuda.empty_cache(); log("7B directo")
        except Exception as ex: print("  (aviso) 7B no disponible:", repr(ex)[:160])
crit = dano*(0.5+0.5*grav)

# ------------------------------- alumnos baratos -------------------------------
def alumno(Xtr, ytr, Xte):
    """MLP 64 como la sonda del paper 1; objetivo blando en [0,1]. Devuelve la nota para Xte."""
    if SIMULACRO:
        from sklearn.linear_model import Ridge
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6; return Ridge(alpha=10).fit((Xtr-mu)/sd, ytr).predict((Xte-mu)/sd)
    torch.manual_seed(0); X = torch.tensor(Xtr, dtype=torch.float32, device="cuda"); y = torch.tensor(ytr, dtype=torch.float32, device="cuda")
    mu, sd = X.mean(0), X.std(0)+1e-6
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(net((X-mu)/sd).squeeze(-1), y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.sigmoid(net((torch.tensor(Xte, dtype=torch.float32, device="cuda")-mu)/sd).squeeze(-1)).cpu().numpy()
A = slice(nK+nE, M); u = lambda v: (v-v.min())/(v.max()-v.min()+1e-9)
PUNTOS = {"SUEÑO (daño x gravedad)": (crit, "maestro"), "DIRECTO juez 1.5B": (dir15, "maestro"),
          "sueño: daño medio de las 8 preguntas": (dmed, "variante"), "sueño: solo influencia": (infl, "variante"),
          "directo 0.5B": (dir05, "aparte"), "e5 coseno al criterio (sin entrenar)": (e5cos, "aparte"),
          "alumno sonda L16 <- sueño": (alumno(XL[A], u(crit[A]), XL), "alumno"), "alumno sonda L16 <- directo": (alumno(XL[A], u(dir15[A]), XL), "alumno"),
          "alumno e5 <- sueño": (alumno(XE[A], u(crit[A]), XE), "alumno"), "alumno e5 <- directo": (alumno(XE[A], u(dir15[A]), XE), "alumno"),
          "sonda L16 con etiquetas humanas": (alumno(XLh, Y_H, XL), "humana"), "e5 con etiquetas humanas": (alumno(XEh, Y_H, XE), "humana")}
if dir7 is not None: PUNTOS["directo 7B"] = (dir7, "aparte")

# ------------------------------- evaluacion -------------------------------
def dif(y, a, b, B=2000):
    r = np.random.default_rng(0); ds = []
    for _ in range(B):
        k = r.integers(0, len(y), len(y))
        if 0 < y[k].sum() < len(k): ds.append(roc_auc_score(y[k], a[k])-roc_auc_score(y[k], b[k]))
    return roc_auc_score(y, a)-roc_auc_score(y, b), np.percentile(ds, 2.5), np.percentile(ds, 97.5)
CSV, RES = [], {}
for nombre, sl in [("kalandor", slice(0, nK)), ("examen", slice(nK, nK+nE))]:
    S = SETS[nombre]; y = S["y"]; k = int(y.sum()); tr = np.array(["trampa" in c for c in S["cat"]])
    print(f"\n================ {nombre.upper()} · n={len(y)} · críticos={k} · trampas={tr.sum()} ================")
    print(f"  {'metodo':40s} {'clase':9s} AUROC  críticos en su top-{k}  trampas en su top-{k}")
    filas = []
    for n, (v, c) in PUNTOS.items():
        s = v[sl]; top = np.argsort(-s)[:k]; filas.append((roc_auc_score(y, s), n, c, int(y[top].sum()), int(tr[top].sum())))
    for a, n, c, ct, tt in sorted(filas, reverse=True):
        print(f"  {n:40s} {c:9s} {a:.3f}  {ct:3d}/{k}                 {tt:3d}/{tr.sum()}"); CSV.append(f"{nombre},{n},{c},{a:.4f},{ct},{tt}")
    RES[nombre] = dif(y, crit[sl], dir15[sl]); d, lo, hi = RES[nombre]
    print(f"  SUEÑO - DIRECTO 1.5B = {d:+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]")
    print("  media por categoría (influencia / daño / gravedad / sueño / directo 1.5B):")
    for g in sorted(set(S["cat"])):
        ii = np.array([i for i, c in enumerate(S["cat"]) if c == g]); o = sl.start
        print(f"    {g:14s} {infl[o+ii].mean():.3f} / {dano[o+ii].mean():.2f} / {grav[o+ii].mean():.2f} / {crit[o+ii].mean():.2f} / {dir15[o+ii].mean():.2f}")
aK = roc_auc_score(SETS["kalandor"]["y"], dir15[:nK]); (dE, loE, hiE), dK = RES["examen"], RES["kalandor"][0]
v = ("INCONCLUSO (el juez directo no llega a 0.60 en Kalandor)" if aK < 0.60 else "SUEÑO GANA" if dE >= 0.05 and loE > 0 and dK > -0.03 else
     "DIRECTO GANA" if dE <= -0.05 and hiE < 0 else "EMPATE" if abs(dE) < 0.03 else "INTERMEDIO")
print("\n================ VEREDICTO (reglas escritas antes de lanzar) ================")
print(f"  {v} · examen: sueño - directo = {dE:+.3f} [{loE:+.3f}, {hiE:+.3f}] · Kalandor: {dK:+.3f}")
print("  prediccion escrita: DIRECTO GANA o EMPATE")
print("\n================ CSV ================\nconjunto,metodo,clase,auroc,criticos_top,trampas_top")
for l in CSV: print(l)
print("\nconjunto,i,critico,categoria,sueño,directo15,directo7,influencia,daño,gravedad")
for nombre, o in [("kalandor", 0), ("examen", nK)]:
    S = SETS[nombre]
    for i in range(len(S["y"])): print(f"{nombre},{i},{S['y'][i]},{S['cat'][i]},{crit[o+i]:.3f},{dir15[o+i]:.3f},{(dir7[o+i] if dir7 is not None else float('nan')):.3f},{infl[o+i]:.3f},{dano[o+i]:.3f},{grav[o+i]:.3f}")
log("FIN")
