import os
import re
import json
from collections import Counter
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import emoji
from anthropic import Anthropic, AuthenticationError, RateLimitError

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)


@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://d3js.org; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "connect-src 'self';"
    )
    return response


@app.route('/')
def index():
    return app.send_static_file('index.html')


# ─── PARSING ────────────────────────────────────────────────────────────────

PATTERNS = [
    # iOS English 12hr:  1/15/23, 3:42 PM - Name: msg
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M)\s+-\s+(.+?):\s+(.+)$', re.I),
    # iOS Spanish 24hr:  15/01/23, 15:42 - Name: msg
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+-\s+(.+?):\s+(.+)$'),
    # Android no comma:  15/01/2023 15:42 - Name: msg
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+-\s+(.+?):\s+(.+)$'),
    # Bracketed:         [15/01/2023, 15:42] Name: msg
    re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?)\]\s+(.+?):\s+(.+)$'),
]

SYSTEM_RE = re.compile(
    r'(?:llamada de voz perdida|llamada de video perdida|llamada iniciada|'
    r'llamada perdida|missed voice call|video call|voice call|'
    r'you were added|messages and calls are end-to-end encrypted|'
    r'mensajes y llamadas|cifrados de extremo|you created group|'
    r'changed the subject|changed this group|added you)',
    re.I
)

CALL_RE = re.compile(
    r'(?:llamada de voz|llamada de video|llamada iniciada|'
    r'missed voice call|video call|voice call)',
    re.I
)

MEDIA_RE = re.compile(
    r'(?:<media omitted>|imagen omitida|video omitido|audio omitido|'
    r'sticker omitido|documento omitido|gif omitido|'
    r'contact card omitted|tarjeta de contacto omitida)',
    re.I
)


def parse_datetime(date_str: str, time_str: str):
    combined = f"{date_str} {time_str.strip()}"
    fmts = [
        '%d/%m/%Y %I:%M %p', '%d/%m/%Y %I:%M:%S %p',
        '%d/%m/%Y %H:%M',    '%d/%m/%Y %H:%M:%S',
        '%d/%m/%y %I:%M %p', '%d/%m/%y %I:%M:%S %p',
        '%d/%m/%y %H:%M',    '%d/%m/%y %H:%M:%S',
        '%m/%d/%Y %I:%M %p', '%m/%d/%y %I:%M %p',
        '%m/%d/%Y %H:%M',    '%m/%d/%y %H:%M',
        '%m/%d/%Y %I:%M:%S %p', '%m/%d/%y %I:%M:%S %p',
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp_chat(text: str):
    # Normalize non-standard whitespace inserted by WhatsApp iOS exports
    # U+202F narrow no-break space appears between time and AM/PM on iPhone
    text = text.replace('\u202f', ' ').replace('\u200e', '').replace('\u200f', '')

    messages = []
    current = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        matched = False
        for pat in PATTERNS:
            m = pat.match(line)
            if m:
                if current:
                    messages.append(current)
                date_s, time_s, sender, content = m.groups()
                dt = parse_datetime(date_s, time_s)
                content = content.strip()
                current = {
                    'datetime': dt,
                    'sender': sender.strip(),
                    'content': content,
                    'is_system': bool(SYSTEM_RE.search(content)),
                    'is_call':   bool(CALL_RE.search(content)),
                    'is_media':  bool(MEDIA_RE.search(content)),
                }
                matched = True
                break

        if not matched and current:
            current['content'] += ' ' + line

    if current:
        messages.append(current)

    return messages


# ─── HELPERS ────────────────────────────────────────────────────────────────

def extract_emojis(text: str):
    return [c for c in text if c in emoji.EMOJI_DATA]


def detect_language(df: pd.DataFrame) -> str:
    es_words = {'hola', 'que', 'como', 'bien', 'gracias', 'bueno', 'pero',
                'porque', 'cuando', 'donde', 'estar', 'tener', 'hacer',
                'decir', 'querer', 'saber', 'poder', 'poner', 'venir'}
    en_words = {'hello', 'what', 'how', 'good', 'thanks', 'because', 'when',
                'where', 'have', 'know', 'want', 'need', 'think', 'come',
                'going', 'been', 'that', 'this', 'with', 'from'}

    sample = ' '.join(
        df[~df['is_system'] & df['content'].notna()]['content'].head(300)
    ).lower()
    words = set(re.findall(r'\b[a-z]{3,}\b', sample))

    es_score = len(words & es_words)
    en_score = len(words & en_words)
    return 'español' if es_score >= en_score else 'inglés'


# ─── LIGHTWEIGHT SPELL CHECKER ───────────────────────────────────────────────
# Replaces pyspellchecker (~60 MB of frequency dicts) with a frozenset.
# Memory: ~3 MB vs ~60 MB.  Startup: instant vs ~8 seconds.
#
# Strategy: a word is NOT an error if:
#   (a) it appears in _BASE_VOCAB (core Spanish + English words), OR
#   (b) it appears 2+ times in the chat itself (probably intentional slang/name).
# Only words that are both rare in the chat AND absent from base vocab are flagged.

def _build_vocab() -> frozenset:
    words = set()

    # Spanish — function words, pronouns, conjunctions, prepositions, adverbs
    words.update("""
    el la los las un una unos unas de del en con por para sin sobre bajo ante
    tras a al y o u e ni pero sino aunque que si no ya muy mas menos tan tanto
    asi bien mal aqui ahi alli hoy ayer manana ahora antes despues tambien
    tampoco solo todo todos nada algo alguien nadie siempre nunca casi cerca
    lejos entonces luego pues claro quizas tal vez incluso ademas mientras
    durante entre hasta desde hacia segun excepto salvo osea eso esa ese esto
    igual mismo misma mismos mismas otro otra otros otras cada cual cuales
    quien quienes donde cuando como cuanto cuanta cuantos cuantas
    apenas aunque porque porqué sino sinó sino acá allá acá tras via
    """.split())

    # Spanish pronouns
    words.update("""
    yo tú él ella nosotros vosotros ellos ellas usted ustedes
    me te se nos os le les lo mi mis tu tus su sus
    nuestro nuestra nuestros nuestras vuestro vuestra
    mio mia tuyo tuya suyo suya
    este esta estos estas ese esa esos esas aquel aquella aquellos aquellas
    """.split())

    # Spanish verbs — infinitives + most common conjugations
    words.update("""
    ser estar tener hacer ir venir poder querer saber ver dar decir llegar
    pasar llevar salir poner sentir creer conocer vivir quedar pensar parecer
    traer perder necesitar esperar buscar usar trabajar empezar entrar recordar
    dejar abrir leer escribir pagar cambiar ganar jugar correr volver comprar
    conseguir ayudar lograr crear caer cerrar incluir servir morir subir bajar
    olvidar terminar comenzar continuar estudiar explicar preguntar responder
    intentar aprender escuchar mirar tocar comer beber dormir despertar caminar
    gritar llorar reir decidir elegir preferir aceptar permitir invitar celebrar
    preparar organizar planear viajar regresar visitar explorar descubrir mejorar
    aumentar parar detener mover limpiar guardar mostrar aparecer funcionar
    depender resultar ocurrir existir seguir hablar llamar sacar acabar contar
    pedir tomar deber nacer crecer entender comprender manejar pasar revisar
    mandar probar jugar salvar ganar perder compartir cumplir avanzar
    soy eres somos sois son era eras éramos eran
    fui fuiste fue fuimos fueron
    seré serás será seremos seréis serán
    sería serías seríamos serían
    sido siendo
    tengo tienes tiene tenemos tenéis tienen
    tenía tenías teníamos tenían
    tuvo tuve tuviste tuvimos tuvieron
    tendré tendrás tendrá tendremos tendréis tendrán
    hago haces hace hacemos hacéis hacen
    hice hiciste hizo hicimos hicieron
    haré harás hará haremos haréis harán
    voy vas va vamos vais van
    fui fuiste fue fuimos fueron
    iré irás irá iremos iréis irán
    vine viniste vino vinimos vinieron
    vengo vienes viene venimos vienen
    vendré vendrás vendrá vendremos vendrán
    puedo puedes puede podemos pueden
    pudo pude pudiste pudimos pudieron
    podré podrás podrá podremos podrán
    quiero quieres quiere queremos quieren
    quiso quisiste quisimos quisieron
    querré querrás querrá querremos querrán
    sé sabes sabe sabemos saben supo supe
    veo ves ve vemos ven vió vi
    doy das da damos dan dio di
    digo dices dice decimos dicen dijo dije dijiste dijimos dijeron
    diré dirás dirá diremos diréis dirán
    hay hubo había habrá habría habría
    estoy estás está estamos estáis están
    estaba estabas estábamos estaban
    estuve estuviste estuvo estuvimos estuvieron
    estaré estarás estará estaremos estarán
    pienso piensas piensa pensamos piensan
    pensé pensaste pensó pensamos pensaron
    puedo puedes puede podemos podemos
    hablo hablas habla hablamos hablan
    hablé hablaste habló hablamos hablaron
    llego llegas llega llegamos llegan
    llegué llegaste llegó llegamos llegaron
    vivo vives vive vivimos viven
    como comes come comemos comen
    sigo sigues sigue seguimos siguen
    quedo quedas queda quedamos quedan
    pongo pones pone ponemos ponen
    salgo sales sale salimos salen
    traigo traes trae traemos traen
    """.split())

    # Spanish nouns — people, places, time, objects, concepts
    words.update("""
    persona personas gente vida tiempo lugar casa trabajo día mes año semana
    hora minuto segundo grupo chat mensaje foto video audio llamada
    amigo amiga amigos amigas familia hijo hija padre madre hermano hermana
    abuela abuelo novio novia esposo esposa compañero compañera
    profesor profesora profe maestro maestra estudiante alumno alumna
    jefe jefa doctor doctora chico chica niño niña adulto señor señora
    chavo chava cuate carnal mano bro wey güey tipo tipa gente
    dinero precio costo gasto
    comida agua café bebida cerveza pizza taco tacos hamburguesa
    pan leche azúcar sal arroz frijoles carne pollo pescado fruta verdura postre
    desayuno almuerzo comida cena snack antojo
    ciudad país estado pueblo colonia calle avenida departamento cuarto sala
    cocina baño jardín parque plaza mercado tienda oficina escuela colegio
    universidad hospital iglesia restaurante bar hotel aeropuerto estación
    carretera camino carro auto autobús metro tren avión moto bici
    ropa camisa pantalón vestido zapatos tenis calcetines playera sudadera
    libro cuaderno lápiz pluma computadora laptop teléfono celular pantalla
    internet red aplicación app juego película canción música álbum artista
    cosa forma tipo parte punto lado razón causa efecto resultado idea
    opinión tema asunto problema solución plan proyecto esfuerzo apoyo cambio
    verdad mentira historia cuento chiste broma pregunta respuesta noticia
    examen tarea exposición calificación nota materia semestre período
    partido jugador equipo gol punto cancha campo estadio torneo campeonato
    boleto entrada concierto evento fiesta reunión cita viaje destino
    clase práctica trabajo proyecto entrega fechas
    foto imagen video clip meme sticker reacción
    """.split())

    # Spanish adjectives
    words.update("""
    bueno buena buenos buenas malo mala malos malas grande pequeño alto bajo
    largo corto gordo flaco delgado bonito feo guapo guapa lindo linda
    rápido lento fácil difícil importante necesario posible imposible verdadero
    falso seguro peligroso libre ocupado cansado feliz triste enojado asustado
    emocionado sorprendido aburrido enfermo sano rico pobre fuerte débil
    nuevo viejo joven mayor menor primero último siguiente anterior próximo
    diferente igual mismo distinto especial normal extraño raro increíble
    perfecto imperfecto interesante aburrido divertido chistoso gracioso
    serio tranquilo nervioso ansioso estresado relajado contento alegre
    genial chido chida padre cool pesado intenso heavy brutal
    """.split())

    # Spanish greetings, exclamations, discourse markers
    words.update("""
    hola buenas buenos días tardes noches bienvenido bienvenida adiós chao
    gracias por favor perdona disculpa salud felicidades suerte ánimo
    vale venga dale vamos anda ven mira oye escucha sigue espera para claro
    exacto obvio obvio correcto incorrecto cierto verdad mentira
    wow ohhh mmm ahh ohh ugh ehh umm pues sea ósea osea oki
    """.split())

    # Spanish numbers and calendar
    words.update("""
    uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece
    catorce quince dieciséis diecisiete dieciocho diecinueve veinte
    treinta cuarenta cincuenta sesenta setenta ochenta noventa cien
    doscientos trescientos mil millón primero segundo tercero cuarto quinto
    lunes martes miércoles jueves viernes sábado domingo
    enero febrero marzo abril mayo junio julio agosto septiembre octubre
    noviembre diciembre
    """.split())

    # More verb conjugations — preterite and common forms that get flagged
    words.update("""
    perdí perdiste perdió perdimos perdieron
    gané ganaste ganó ganamos ganaron
    aprendí aprendiste aprendió aprendimos aprendieron
    comí comiste comió comimos comieron
    bebí bebiste bebió bebimos bebieron
    dormí dormiste durmió dormimos durmieron
    salí saliste salió salimos salieron
    llegué llegaste llegó llegamos llegaron
    entré entraste entró entramos entraron
    empecé empezaste empezó empezamos empezaron
    terminé terminaste terminó terminamos terminaron
    estudié estudiaste estudió estudiamos estudiaron
    trabajé trabajaste trabajó trabajamos trabajaron
    compré compraste compró compramos compraron
    jugué jugaste jugó jugamos jugaron
    corrí corriste corrió corrimos corrieron
    viví viviste vivió vivimos vivieron
    sentí sentiste sintió sentimos sintieron
    pedí pediste pidió pedimos pidieron
    volví volviste volvió volvimos volvieron
    subí subiste subió subimos subieron
    bajé bajaste bajó bajamos bajaron
    conocí conociste conoció conocimos conocieron
    saqué sacaste sacó sacamos sacaron
    metí metiste metió metimos metieron
    llamé llamaste llamó llamamos llamaron
    mandé mandaste mandó mandamos mandaron
    pagué pagaste pagó pagamos pagaron
    cambié cambiaste cambió cambiamos cambiaron
    moví moviste movió movimos movieron
    abrí abriste abrió abrimos abrieron
    cerré cerraste cerró cerramos cerraron
    leí leíste leyó leímos leyeron
    escribí escribiste escribió escribimos escribieron
    busqué buscaste buscó buscamos buscaron
    encontré encontraste encontró encontramos encontraron
    dejé dejaste dejó dejamos dejaron
    probé probaste probó probamos probaron
    pasé pasaste pasó pasamos pasaron
    traté trataste trató tratamos trataron
    ayudé ayudaste ayudó ayudamos ayudaron
    viajé viajaste viajó viajamos viajaron
    regresé regresaste regresó regresamos regresaron
    invité invitaste invitó invitamos invitaron
    celebré celebraste celebró celebramos celebraron
    recordé recordaste recordó recordamos recordaron
    olvidé olvidaste olvidó olvidamos olvidaron
    pregunté preguntaste preguntó preguntamos preguntaron
    respondí respondiste respondió respondimos respondieron
    expliqué explicaste explicó explicamos explicaron
    decidi decidiste decidió decidimos decidieron
    compartí compartiste compartió compartimos compartieron
    """.split())

    # Common adverbs and time expressions
    words.update("""
    anoche antier antayer temprano tarde pronto próximamente recientemente
    finalmente básicamente generalmente normalmente especialmente
    definitivamente probablemente posiblemente seguramente exactamente
    perfectamente fácilmente rápidamente lentamente claramente obviamente
    directamente absolutamente completamente totalmente únicamente
    realmente verdaderamente simplemente solamente únicamente prácticamente
    """.split())

    # Common nouns missed
    words.update("""
    crimen milagro momento detalle recuerdo chisme rumor sorpresa noticia
    susto accidente problema solución situación condición conexión acción
    reacción emoción sensación intención decisión descripción explicación
    información comunicación conversación discusión argumento comentario
    opinión posición versión función aplicación operación dirección
    historia imagen cuenta número lista mensaje respuesta pregunta idea
    concepto significado sentido razón motivo objetivo propósito resultado
    beneficio ventaja desventaja riesgo oportunidad posibilidad probabilidad
    realidad verdad mentira fantasía sueño pesadilla recuerdo olvido
    inicio final principio final centro medio borde límite nivel grado
    tipo clase especie forma manera modo estilo calidad cantidad valor
    tamaño peso altura velocidad temperatura color sonido sabor olor
    partido juego match torneo competencia clasificación semifinal final
    gol autogol penalti penalty árbitro jugador portero defensa mediocampo
    aficionado hincha estadio grada tribuna cancha campo terreno
    """.split())

    # Common adjectives missed
    words.update("""
    histórico único famoso popular conocido desconocido complicado sencillo
    simple complejo completo incompleto disponible imposible interesante
    aburrido emocionante divertido gracioso chistoso serio formal informal
    oficial particular personal profesional académico económico político
    social cultural deportivo artístico musical cinematográfico
    increíble impresionante espectacular extraordinario maravilloso
    terrible horrible espantoso aterrador decepcionante frustrante
    cansado agotado activo pasivo positivo negativo neutral
    moderno antiguo clásico contemporáneo actual reciente futuro pasado
    real virtual físico digital local global nacional internacional
    """.split())

    # Common reflexive and compound verb forms
    words.update("""
    enseñarte explicarte ayudarte esperarte llamarte mandarte
    contarte decirte pedirte invitarte llevarte traerte
    enseñarme explicarme ayudarme llamarme mandarte
    pasándolo haciéndolo diciéndolo viéndolo
    estamos están estuve estuviste estuvo estuvimos estuvieron
    """.split())

    # Common words frequently missed — cinema, conditional, slang, informal expressions
    words.update("""
    cine teatro concierto evento festival espectáculo actuación presentación
    podría podrías podríamos podrían
    querría querrías querríamos querrían
    debería deberías deberíamos deberían
    tendría tendrías tendríamos tendrían
    habría habrías habríamos habrían
    sería serías seríamos serían
    checa checar chécalo chécala
    oigan órale ándale híjole caray
    neta mero mera chamba pedo gana
    carísimo baratísimo facilísimo difícilísimo
    tranquilo tranquila tranquilos tranquilas
    genial brutal increíble emocionante divertido divertida
    perfecto perfecta perfectos perfectas
    candidato candidata candidatos candidatas
    promedio calificaciones promedios boletas
    práctica prácticas entrega entregas fecha fechas
    whatsapp instagram facebook twitter tiktok youtube
    """.split())

    # English — core vocabulary for mixed-language chats
    words.update("""
    the and are was were been have has had will would could should may might
    that this with from they them their there then than those these
    you your him her our out some more very also just one all but get did
    for not its say see use let say yes okay yes not who how what why when
    where which can your him her our its will would could should
    hello goodbye thanks please sorry good bad great nice cool wow
    like love hate want need know think feel make take give find keep put
    come see look try ask work help start stop move turn buy sell pay
    call text send show tell remember believe understand decide choose
    today tomorrow yesterday morning afternoon evening night week month year
    time moment always never sometimes often usually
    friend family brother sister mom dad parent child baby girl man woman
    person people group team class school homework test project teacher
    student food coffee drink water beer breakfast lunch dinner
    phone video picture movie music song game
    actually basically really totally definitely probably maybe sure right
    going getting coming done ready wait hold let
    """.split())

    # Normalize accented → plain for lookup flexibility
    accented = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'
    }
    stripped_words = set()
    for w in words:
        sw = w
        for a, b in accented.items():
            sw = sw.replace(a, b)
        stripped_words.add(sw)

    return frozenset(words | stripped_words)


_BASE_VOCAB: frozenset = _build_vocab()


def calculate_spelling_errors(df: pd.DataFrame, members: list):
    ignore_re = re.compile(
        r'^(\d+|[a-z]|https?|www|jaja+|jeje+|haha+|lol|xd|ok|'
        r'q|x|k|tb|tmb|pq|xq|wey|bro|si|no|ja|je|bye|omg|wow)$', re.I
    )

    stats = {m: {'errors': 0, 'total': 0} for m in members}
    error_freq: Counter = Counter()

    df_sample = df[~df['is_system'] & ~df['is_media']].head(5_000)
    member_words: dict = {m: [] for m in members}
    for _, row in df_sample.iterrows():
        if not isinstance(row['content'], str) or row['sender'] not in member_words:
            continue
        words = [w for w in re.findall(r"\b[a-záéíóúüñ']{3,}\b", row['content'].lower())
                 if not ignore_re.match(w)]
        member_words[row['sender']].extend(words)

    # Words used 2+ times across the entire chat are considered intentional
    # (slang, names, abbreviations users repeat on purpose)
    global_freq: Counter = Counter(w for ws in member_words.values() for w in ws)
    chat_vocab = {w for w, c in global_freq.items() if c >= 2}

    # A word is an error only if it's absent from the base vocab AND rare in this chat
    for member, words in member_words.items():
        for word in words:
            stats[member]['total'] += 1
            if word not in _BASE_VOCAB and word not in chat_vocab:
                stats[member]['errors'] += 1
                error_freq[word] += 1

    ranking = sorted(
        [{'member': m,
          'errors': s['errors'],
          'total_words': s['total'],
          'error_rate': round(s['errors'] / s['total'] * 100, 2) if s['total'] else 0}
         for m, s in stats.items()],
        key=lambda x: x['error_rate'], reverse=True
    )

    frequent_errors = [
        {'word': word, 'count': cnt}
        for word, cnt in error_freq.most_common(20)
    ]

    word_cloud_data = [
        {'text': word, 'weight': cnt}
        for word, cnt in error_freq.most_common(60)
    ]

    return ranking, frequent_errors, word_cloud_data


def detect_bursts(df: pd.DataFrame, window_min=30, min_msgs=12):
    valid = df[~df['is_system']].dropna(subset=['datetime']).sort_values('datetime')
    bursts = []
    i = 0
    rows = valid.reset_index(drop=True)
    while i < len(rows):
        t0 = rows.iloc[i]['datetime']
        t1 = t0 + timedelta(minutes=window_min)
        window = rows[(rows['datetime'] >= t0) & (rows['datetime'] <= t1)]
        if len(window) >= min_msgs:
            bursts.append({
                'start': t0.strftime('%d/%m/%Y %H:%M'),
                'message_count': len(window),
                'participants': window['sender'].unique().tolist(),
                'sample': str(window.iloc[0]['content'])[:120],
            })
            i += len(window)
        else:
            i += 1
    return bursts[:20]


def calculate_interaction_network(df: pd.DataFrame, members: list):
    valid = df[~df['is_system']].dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    pairs: Counter = Counter()
    for i in range(1, len(valid)):
        prev, curr = valid.iloc[i - 1], valid.iloc[i]
        if prev['sender'] != curr['sender'] and prev['sender'] in members and curr['sender'] in members:
            diff_min = (curr['datetime'] - prev['datetime']).total_seconds() / 60
            if diff_min <= 15:
                pairs[f"{curr['sender']}→{prev['sender']}"] += 1

    return sorted(
        [{'source': p.split('→')[0], 'target': p.split('→')[1], 'weight': c}
         for p, c in pairs.most_common(40)],
        key=lambda x: x['weight'], reverse=True
    )


def calculate_ghosting(df: pd.DataFrame, members: list):
    valid = df[~df['is_system']].dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    ghost = {m: {'sent': 0, 'ghosted': 0} for m in members}

    for i in range(len(valid) - 1):
        curr, nxt = valid.iloc[i], valid.iloc[i + 1]
        s = curr['sender']
        if s not in ghost:
            continue
        ghost[s]['sent'] += 1
        if curr['sender'] != nxt['sender']:
            diff_h = (nxt['datetime'] - curr['datetime']).total_seconds() / 3600
            if diff_h > 24:
                ghost[s]['ghosted'] += 1

    return sorted(
        [{'member': m,
          'ghosted_count': g['ghosted'],
          'total_sent': g['sent'],
          'ghosting_rate': round(g['ghosted'] / g['sent'] * 100, 2) if g['sent'] else 0}
         for m, g in ghost.items()],
        key=lambda x: x['ghosting_rate'], reverse=True
    )


def calculate_triple_texting(df: pd.DataFrame, members: list):
    valid = df[~df['is_system']].sort_values('datetime').reset_index(drop=True)
    counts: Counter = Counter()
    i = 0
    while i < len(valid):
        sender = valid.iloc[i]['sender']
        run = 1
        while i + run < len(valid) and valid.iloc[i + run]['sender'] == sender:
            run += 1
        if run >= 3 and sender in members:
            counts[sender] += 1
        i += run

    return sorted(
        [{'member': m, 'count': c} for m, c in counts.items()],
        key=lambda x: x['count'], reverse=True
    )


def get_top_ngrams(df: pd.DataFrame, n=2, top_k=10):
    STOP = {
        'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'un', 'una', 'los',
        'las', 'del', 'con', 'por', 'para', 'se', 'no', 'si', 'lo', 'le',
        'su', 'al', 'yo', 'tu', 'me', 'te', 'mi', 'como', 'pero', 'más',
        'ya', 'muy', 'pues', 'bien', 'porque', 'cuando', 'este', 'esta',
        'hay', 'todo', 'así', 'the', 'a', 'an', 'is', 'in', 'it', 'of',
        'to', 'and', 'or', 'but', 'he', 'she', 'we', 'they', 'this', 'that',
        'was', 'are', 'for', 'on', 'with', 'have', 'not', 'be', 'at',
    }

    all_words = []
    for content in df[~df['is_system'] & ~df['is_media']]['content'].dropna():
        ws = [w for w in re.findall(r'\b[a-záéíóúüñ]{3,}\b', content.lower()) if w not in STOP]
        all_words.extend(ws)

    if n == 1:
        counter = Counter(all_words)
    else:
        counter = Counter(
            ' '.join(all_words[i:i + n]) for i in range(len(all_words) - n + 1)
        )

    return [{'phrase': ph, 'count': c} for ph, c in counter.most_common(top_k)]


# ─── CLAUDE AI ──────────────────────────────────────────────────────────────

def call_claude_api(api_key: str, stats: dict, language: str) -> dict:
    """Returns parsed AI result dict, or raises with a reason string attached as .reason."""
    client = Anthropic(api_key=api_key)

    lang_instr = (
        "Responde ÚNICAMENTE en español."
        if language == 'español'
        else "Respond ONLY in English."
    )

    prompt = f"""Analiza las siguientes estadísticas de un chat de WhatsApp.

{lang_instr}

Estadísticas:
{json.dumps(stats, ensure_ascii=False, indent=2)}

Devuelve ÚNICAMENTE JSON válido (sin backticks ni texto extra) con esta estructura:
{{
  "analisis_vibe": "Análisis narrativo y creativo de 3-4 oraciones sobre la dinámica del grupo.",
  "mapeo_temas": [
    {{"nombre": "Nombre del tema", "descripcion": "Descripción breve"}}
  ],
  "perfiles_personales": {{
    "NombreMiembro": {{"descripcion": "Estilo de comunicación en 1-2 oraciones.", "rol": "Rol en el grupo"}}
  }}
}}

Incluye al menos 3 temas y perfiles para cada miembro listado."""

    try:
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=2048,
            messages=[{'role': 'user', 'content': prompt}],
        )
    except AuthenticationError as exc:
        exc.reason = 'invalid_key'
        raise
    except RateLimitError as exc:
        exc.reason = 'quota_exceeded'
        raise
    except Exception as exc:
        exc.reason = 'network_error'
        raise

    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ─── MAIN ENDPOINT ──────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Falta el archivo de chat'}), 400

    file = request.files['file']
    api_key = request.form.get('api_key', '').strip()

    if not file or file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    # Read file — try encodings in order; WhatsApp exports vary by platform/region
    try:
        raw = file.read()
        content = None
        for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            content = raw.decode('latin-1', errors='replace')
    except Exception as exc:
        return jsonify({'error': f'Error leyendo el archivo: {exc}'}), 400

    # Parse
    raw_msgs = parse_whatsapp_chat(content)
    if not raw_msgs:
        return jsonify({'error': 'No se encontraron mensajes en el archivo'}), 400

    # Truncate
    real_total = len(raw_msgs)
    truncated = real_total > 50_000
    if truncated:
        raw_msgs = raw_msgs[-50_000:]

    # DataFrame — use categorical dtype for sender to cut string memory ~4x
    df = pd.DataFrame(raw_msgs)
    df = df.dropna(subset=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['sender'] = df['sender'].astype('category')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    df['quarter'] = df['datetime'].dt.to_period('Q').astype(str)

    # Members: require at least 1 real text message (excludes system/media-only senders).
    # Use astype(str) so categorical dtype doesn't include zero-count categories.
    df_text = df[~df['is_system'] & ~df['is_media']]
    vc = df_text['sender'].astype(str).value_counts()
    members = [m for m in vc[vc > 0].index if m and len(m) <= 50][:15]

    if not members:
        return jsonify({'error': 'No se encontraron participantes reales en el chat. Verifica que el archivo sea una exportación válida de WhatsApp.'}), 400

    df_real = df[df['sender'].isin(members) & ~df['is_system']].copy()

    # ── Metadata ──────────────────────────────────────────────────────────
    total_messages = len(df_real)
    total_words = sum(
        len(re.findall(r'\w+', str(c)))
        for c in df_real[~df_real['is_media']]['content']
    )

    all_emojis: list = []
    for c in df_real['content'].dropna():
        all_emojis.extend(extract_emojis(str(c)))
    total_emojis = len(all_emojis)
    total_media = int(df['is_media'].sum())
    total_calls = int(df['is_call'].sum())

    date_range = ''
    if len(df_real):
        date_range = (
            f"{df_real['datetime'].min().strftime('%d/%m/%Y')} – "
            f"{df_real['datetime'].max().strftime('%d/%m/%Y')}"
        )

    # ── Activity ──────────────────────────────────────────────────────────
    by_month = (
        df_real.groupby('month').size()
        .reset_index(name='count')
        .sort_values('month')
    )
    by_month_list = [{'month': r['month'], 'count': int(r['count'])}
                     for _, r in by_month.iterrows()]

    heatmap_raw = (
        df_real.groupby(['hour', 'day_of_week']).size()
        .reset_index(name='count')
    )
    heatmap_list = [
        {'hour': int(r['hour']), 'day': int(r['day_of_week']), 'count': int(r['count'])}
        for _, r in heatmap_raw.iterrows()
    ]

    # ── Members ───────────────────────────────────────────────────────────
    quarterly_raw = (
        df_real.groupby(['quarter', 'sender']).size()
        .reset_index(name='count')
        .sort_values('quarter')
    )
    quarterly_list = [
        {'quarter': r['quarter'], 'member': r['sender'], 'count': int(r['count'])}
        for _, r in quarterly_raw.iterrows()
    ]

    lexical_richness = []
    for member in members:
        texts = df_real[df_real['sender'] == member]['content'].dropna()
        all_w = re.findall(r'\b[a-záéíóúüñ]{3,}\b', ' '.join(texts).lower())
        total_w = len(all_w)
        if total_w > 0:
            lexical_richness.append({
                'member': member,
                'total_words': total_w,
                'unique_words': len(set(all_w)),
                'richness': round(len(set(all_w)) / total_w, 3),
            })

    interaction_network = calculate_interaction_network(df_real, members)

    # ── Spelling ──────────────────────────────────────────────────────────
    spell_ranking, frequent_errors, word_cloud_data = calculate_spelling_errors(df_real, members)

    # ── Bursts ────────────────────────────────────────────────────────────
    detected_bursts = detect_bursts(df_real)
    top_phrases = get_top_ngrams(df_real, n=2, top_k=10)

    emoji_counter = Counter(all_emojis)
    top_emojis = [{'emoji': e, 'count': c} for e, c in emoji_counter.most_common(10)]

    # ── Dynamics ──────────────────────────────────────────────────────────
    ghosting_rate = calculate_ghosting(df_real, members)
    triple_texting = calculate_triple_texting(df_real, members)

    # ── AI ────────────────────────────────────────────────────────────────
    language = detect_language(df_real)
    stats_summary = {
        'total_messages': total_messages,
        'members': members,
        'date_range': date_range,
        'language': language,
        'top_phrases': top_phrases[:5],
        'most_active': members[0] if members else None,
        'lexical_richness': lexical_richness[:5],
        'ghosting': ghosting_rate[:3],
        'top_emojis': top_emojis[:5],
        'burst_count': len(detected_bursts),
    }

    ai_available = False
    ai_result = None
    ai_error = None
    if len(api_key) > 10:
        try:
            ai_result = call_claude_api(api_key, stats_summary, language)
            ai_available = True
        except Exception as exc:
            ai_error = getattr(exc, 'reason', 'network_error')

    return jsonify({
        'metadata': {
            'total_messages': total_messages,
            'total_words': total_words,
            'total_emojis': total_emojis,
            'total_media': total_media,
            'total_calls': total_calls,
            'date_range': date_range,
            'truncated': truncated,
            'real_total_messages': real_total,
            'members': members,
        },
        'activity': {
            'by_month': by_month_list,
            'heatmap': heatmap_list,
        },
        'members': {
            'quarterly_evolution': quarterly_list,
            'lexical_richness': lexical_richness,
            'interaction_network': interaction_network,
        },
        'spelling': {
            'ranking': spell_ranking,
            'frequent_errors': frequent_errors,
            'word_cloud_data': word_cloud_data,
        },
        'bursts': {
            'detected': detected_bursts,
            'top_phrases': top_phrases,
            'top_emojis': top_emojis,
        },
        'dynamics': {
            'ghosting_rate': ghosting_rate,
            'triple_texting': triple_texting,
        },
        'ai': {
            'available': ai_available,
            'error': ai_error,
            'analisis_vibe': (ai_result or {}).get('analisis_vibe', ''),
            'mapeo_temas':   (ai_result or {}).get('mapeo_temas', []),
            'perfiles_personales': (ai_result or {}).get('perfiles_personales', {}),
        },
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
