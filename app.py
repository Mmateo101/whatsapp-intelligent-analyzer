import os
import re
import json
import io
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import emoji
from spellchecker import SpellChecker
from anthropic import Anthropic

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


def calculate_spelling_errors(df: pd.DataFrame, members: list):
    spell_es = SpellChecker(language='es')
    spell_en = SpellChecker(language='en')

    ignore_re = re.compile(
        r'^(\d+|[a-z]|https?|www|jaja+|jeje+|haha+|lol|xd|ok|'
        r'q|x|k|tb|tmb|pq|xq|wey|bro|si|no|ja|je)$', re.I
    )

    stats = {m: {'errors': 0, 'total': 0} for m in members}
    error_freq: Counter = Counter()

    for _, row in df[~df['is_system'] & ~df['is_media']].iterrows():
        if not isinstance(row['content'], str) or row['sender'] not in stats:
            continue
        words = re.findall(r"\b[a-záéíóúüñ']{3,}\b", row['content'].lower())
        for word in words:
            if ignore_re.match(word):
                continue
            stats[row['sender']]['total'] += 1
            unknown_es = spell_es.unknown([word])
            unknown_en = spell_en.unknown([word])
            if unknown_es and unknown_en:
                suggestion = spell_es.correction(word)
                if suggestion and suggestion != word:
                    stats[row['sender']]['errors'] += 1
                    error_freq[f"{word}→{suggestion}"] += 1

    ranking = sorted(
        [{'member': m,
          'errors': s['errors'],
          'total_words': s['total'],
          'error_rate': round(s['errors'] / s['total'] * 100, 2) if s['total'] else 0}
         for m, s in stats.items()],
        key=lambda x: x['error_rate'], reverse=True
    )

    frequent_errors = [
        {'word': pair.split('→')[0], 'suggestion': pair.split('→')[1], 'count': cnt}
        for pair, cnt in error_freq.most_common(20)
    ]

    word_cloud_data = [
        {'text': pair.split('→')[0], 'weight': cnt}
        for pair, cnt in error_freq.most_common(60)
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

    msg = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=2048,
        messages=[{'role': 'user', 'content': prompt}],
    )
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
    if 'file' not in request.files or 'api_key' not in request.form:
        return jsonify({'error': 'Faltan campos requeridos: file y api_key'}), 400

    file = request.files['file']
    api_key = request.form['api_key'].strip()

    if not file or file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    # Read file
    try:
        content = file.read().decode('utf-8', errors='ignore')
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

    # DataFrame
    df = pd.DataFrame(raw_msgs)
    df = df.dropna(subset=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    df['quarter'] = df['datetime'].dt.to_period('Q').astype(str)

    # Members (top 15 real participants)
    members = [
        m for m in df[~df['is_system']]['sender'].value_counts().index
        if m and len(m) <= 50
    ][:15]

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
    if len(api_key) > 10:
        try:
            ai_result = call_claude_api(api_key, stats_summary, language)
            ai_available = True
        except Exception:
            ai_available = False

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
            'analisis_vibe': (ai_result or {}).get('analisis_vibe', ''),
            'mapeo_temas':   (ai_result or {}).get('mapeo_temas', []),
            'perfiles_personales': (ai_result or {}).get('perfiles_personales', {}),
        },
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
