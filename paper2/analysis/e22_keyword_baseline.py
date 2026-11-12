# E22: selector de palabras clave (lista y regla en e22_palabras_clave_prerregistro.md, fijadas antes de calcular)
import random, re, unicodedata, numpy as np
from sklearn.metrics import roc_auc_score
def limpiar(t):
    t = unicodedata.normalize("NFD", t.lower()); return "".join(c for c in t if unicodedata.category(c) != "Mn")
LISTA = """alergia alergico alergica allergy allergic medicamento medicina medication medicine pastilla pill dosis dose medico doctor
hospital cirugia surgery operacion operation tratamiento treatment enfermedad illness disease diagnostico diagnosis receta prescription
sangre blood salud health cita appointment peligro peligroso danger dangerous seguridad safety emergencia emergency accidente accident
fuego fire gas veneno poison alarma alarm banco bank cuenta account tarjeta card contrasena password pin codigo code deuda debt prestamo
loan pago payment pagar pay factura invoice bill impuesto tax hipoteca mortgage multa fine juicio juez court judge tribunal audiencia
hearing abogado lawyer legal ley law demanda lawsuit recurso appeal pasaporte passport visado visa permiso permit residencia residence
contrato contract plazo deadline vence expires expire renovar renew urgente urgent importante important obligatorio obligatoria mandatory""".split()
PAT = [re.compile(r"(?<![0-9a-z])" + w + r"(s|es)?(?![0-9a-z])") for w in LISTA]
H = [l.rstrip("\n").split("|") for l in open("../data/facts_150.txt", encoding="utf8") if l.strip() and not l.startswith("#")]
Y = np.array([int(h[1]) for h in H]); CAT = [h[2] for h in H]
sc = np.array([sum(len(p.findall(limpiar(h[4]))) > 0 for p in PAT) for h in H])
orden = list(range(150)); random.Random(21).shuffle(orden); NOCHES = [orden[i:i+10] for i in range(0, 150, 10)]
nocrit = [i for i in range(150) if Y[i] == 0]
def auc(idx): return roc_auc_score([Y[i] for i in idx + nocrit], [sc[i] for i in idx + nocrit])
print(f"AUROC global {roc_auc_score(Y, sc):.3f} | explícitos vs no críticos {auc([i for i in range(150) if CAT[i]=='explicito']):.3f} | implícitos vs no críticos {auc([i for i in range(150) if CAT[i]=='implicito']):.3f}")
for c in ["explicito", "implicito", "trampa", "trivial"]:
    v = [sc[i] for i in range(150) if CAT[i] == c]; print(f"   {c:9s}: media {np.mean(v):.2f} · con alguna coincidencia {sum(x>0 for x in v)}/{len(v)}")
L = sum(sum(CAT[i] == "implicito" for i in n) for n in NOCHES if sum(Y[i] for i in n) <= 3)
print(f"Cota inferior de implícitos del juez L = {L}; techo de implícitos con 3 por noche = {sum(min(3, sum(Y[i] for i in n)) for n in NOCHES)} críticos")
res = []
for s in range(5):
    r = random.Random(s); sal = []
    for n in NOCHES: sal += sorted(n, key=lambda i: (-sc[i], r.random()))[:3]
    res.append((sum(Y[i] for i in sal), sum(CAT[i]=="implicito" for i in sal), sum(CAT[i]=="explicito" for i in sal), sum(CAT[i]=="trampa" for i in sal)))
    print(f"desempate {s}: salientes críticos {res[-1][0]}/45 (impl {res[-1][1]}, expl {res[-1][2]}) · trampas {res[-1][3]}")
c, im, ex, tr = np.mean(res, axis=0)
print(f"MEDIA: críticos {c:.1f} · implícitos {im:.1f} · explícitos {ex:.1f} · trampas {tr:.1f}   (juez: 30 críticos, 0 trampas, implícitos >= {L})")
gana = c <= 24 or tr >= 3 or im <= L - 4; empate = c >= 27 and tr <= 1 and im >= L - 1
print("VEREDICTO:", "EL JUEZ GANA" if gana else "EMPATE" if empate else "INTERMEDIO")
