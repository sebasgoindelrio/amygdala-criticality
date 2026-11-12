# E22 · Selector de palabras clave — lista y regla fijadas ANTES de calcular (12 nov 2026, 15:05 CET)

Escrito por Claude antes de ejecutar ningún cálculo con este selector. Sesgo declarado: Claude y el autor ya habían leído
las 150 frases (y Claude escribió sus preguntas y claves). Para limitarlo, la lista se construye solo a partir de las
cuatro categorías de la pregunta del juez (salud, seguridad, dinero, legal u oficial) más palabras genéricas de
urgencia, sin consultar el archivo de hechos al escribirla. No se modificará después de ver los resultados.

## Lista (español / inglés; se acepta el plural añadiendo -s o -es)

- **Salud:** alergia, alergico, alergica, allergy, allergic, medicamento, medicina, medication, medicine, pastilla, pill,
  dosis, dose, medico, doctor, hospital, cirugia, surgery, operacion, operation, tratamiento, treatment, enfermedad,
  illness, disease, diagnostico, diagnosis, receta, prescription, sangre, blood, salud, health, cita, appointment
- **Seguridad:** peligro, peligroso, danger, dangerous, seguridad, safety, emergencia, emergency, accidente, accident,
  fuego, fire, gas, veneno, poison, alarma, alarm
- **Dinero:** banco, bank, cuenta, account, tarjeta, card, contrasena, password, pin, codigo, code, deuda, debt,
  prestamo, loan, pago, payment, pagar, pay, factura, invoice, bill, impuesto, tax, hipoteca, mortgage, multa, fine
- **Legal u oficial:** juicio, juez, court, judge, tribunal, audiencia, hearing, abogado, lawyer, legal, ley, law,
  demanda, lawsuit, recurso, appeal, pasaporte, passport, visado, visa, permiso, permit, residencia, residence,
  contrato, contract, plazo, deadline, vence, expires, expire, renovar, renew
- **Urgencia:** urgente, urgent, importante, important, obligatorio, obligatoria, mandatory

## Puntuación y selección

- Texto de la frase en minúsculas y sin acentos; cada palabra de la lista cuenta como coincidencia si aparece como
  palabra completa (opcionalmente seguida de -s o -es). Puntuación = número de coincidencias.
- Mismo protocolo de salientes que el juez: mismo orden de noches (semilla 21), los 3 hechos nuevos con mayor
  puntuación de cada noche. Empates rotos al azar; se repite con 5 semillas de desempate (0-4) y se reporta la media.
- Sin GPU y sin entrenamiento.

## Qué se reporta

AUROC global (críticos contra no críticos), AUROC explícitos contra no críticos, AUROC implícitos contra no críticos;
entre los 45 salientes: críticos, implícitos, explícitos y trampas. Referencia del juez: 30 críticos, 0 trampas.
Como el log no guarda qué críticos eligió el juez, para los implícitos se usa una cota inferior L del juez: los
implícitos de las noches con 3 críticos o menos, que el juez tuvo que marcar todos para llegar a 30.

## Regla (escrita antes de calcular)

- **EL JUEZ GANA** si se cumple al menos una: palabras clave con ≤ 24 críticos entre sus salientes (media de las 5
  semillas de desempate); ≥ 3 trampas entre sus salientes; ≤ L − 4 implícitos entre sus salientes.
- **EMPATE (haría falta correr las noches con GPU):** ≥ 27 críticos, ≤ 1 trampa y ≥ L − 1 implícitos.
- **INTERMEDIO:** lo demás.
- El resultado se reporta en el paper sea cual sea, incluido INTERMEDIO.

## Resultado (calculado después de guardar lo anterior, 12 nov 2026)

`e22_palabras_clave.py`. AUROC 0.751 (explícitos 0.844, implícitos 0.648; juez 0.992). Entre los 45 salientes (media de
5 desempates): 20.8 críticos (juez 30), 7.6 implícitos (cota del juez L = 10), 13.2 explícitos, 12.0 trampas (juez 0).
**VEREDICTO: EL JUEZ GANA** por dos de los tres criterios (críticos ≤ 24; trampas ≥ 3). El de implícitos no se cumple
(7.6 > L − 4 = 6). No se corren las noches. Reportado en el paper 2, sección 4.1.

## Recuento exacto del juez (después del veredicto, 12 nov 2026, 15:10)

`e22b_implicitos_del_juez.py` en Colab (mismo juez y mismo código, sin entrenar): salientes del juez 30/45 críticos,
15 implícitos, 15 explícitos, 0 trampas. Frente a 7.6 implícitos de las palabras clave. Se reporta como observación
posterior; la regla escrita usaba la cota L = 10 y su veredicto (EL JUEZ GANA, 2 de 3 criterios) no cambia.
