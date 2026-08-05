"""
Hoja Clínica — Backend
-----------------------
API en Flask que recibe una foto de una planta, la envía a la API de
Google Gemini para diagnosticar plagas/enfermedades, y devuelve
un JSON estructurado con el diagnóstico y el plan de tratamiento.

Además incluye /api/tts: convierte el texto del diagnóstico en un audio
con voz natural (Gemini TTS), pensado para agricultores que no saben leer.

Mantener la llamada a la IA en el backend (en vez de en el navegador)
evita exponer la clave de API al público y permite validar, reintentar
y registrar errores de forma centralizada.

La clave de API es GRATIS (con límites generosos) en:
https://aistudio.google.com/apikey
"""

import base64
import json
import logging
import os
import time
import wave
import io

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import google.generativeai as genai

# google-genai es el SDK nuevo, necesario para el modelo de texto-a-voz.
# Instalar con: pip install google-genai
from google import genai as genai_client

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("hoja-clinica")

APP_PORT = int(os.environ.get("PORT", 5000))

GEMINI_API_KEY = "AQ.Ab8RN6KYhAf1JWYXW2MzB1_G7W-YR5kz47DJrwQATECzeP3WxQ"

MODEL_NAME = "gemini-2.5-flash"
TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"
# Voz cálida y clara, apropiada para instrucciones agrícolas en español.
# Otras opciones disponibles: Kore (firme), Achird (amigable), Puck (animado).
TTS_VOICE = "Sulafat"

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB, límite razonable de subida
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_RETRIES = 2
MAX_TTS_CHARS = 2000  # protege contra textos demasiado largos

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY no está configurada. Consíguela gratis en "
        "https://aistudio.google.com/apikey"
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)  # permite que el frontend (otro origen) llame a esta API


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


model = genai.GenerativeModel(MODEL_NAME) if GEMINI_API_KEY else None
tts_client = genai_client.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

DIAGNOSIS_PROMPT = """Eres un ingeniero agrónomo experto en fitosanidad y control de plagas.
Observa la foto de la planta y responde ÚNICAMENTE con un objeto JSON válido,
sin texto adicional, sin explicaciones, sin markdown. Usa exactamente esta forma:

{
  "planta_identificada": "nombre común de la planta si es identificable, o 'planta no identificada'",
  "plaga_o_problema": "nombre de la plaga, enfermedad o problema detectado",
  "severidad": "alta" | "media" | "baja",
  "confianza": "breve frase sobre qué tan clara es la evidencia visual en la foto",
  "sintomas_observados": ["síntoma 1", "síntoma 2"],
  "pasos": ["paso 1 de tratamiento", "paso 2", "paso 3", "paso 4 opcional"],
  "prevencion": "una recomendación breve para evitar que vuelva a ocurrir",
  "urgencia": "si requiere atención inmediata o puede esperar, en una frase"
}

Si la imagen no muestra una planta o no se aprecia ninguna plaga o enfermedad,
usa "plaga_o_problema": "No se detectó plaga visible" y ajusta pasos y
sintomas_observados a cuidados generales de mantenimiento."""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def error_response(message, status=400, code="bad_request"):
    logger.warning("Error %s (%s): %s", status, code, message)
    return jsonify({"ok": False, "error": message, "code": code}), status


def extract_json(text):
    """Limpia la respuesta del modelo y la convierte en dict, con tolerancia
    a que el modelo agregue texto o cercas de markdown por accidente."""
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        preview = cleaned[:300] if cleaned else "(vacío)"
        raise ValueError(
            f"La respuesta del modelo no contiene un JSON válido. Texto recibido: {preview!r}"
        )

    return json.loads(cleaned[start : end + 1])


def call_gemini_with_retries(image_b64, media_type):
    last_error = None
    image_bytes = base64.b64decode(image_b64)

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = model.generate_content(
                [
                    {"mime_type": media_type, "data": image_bytes},
                    DIAGNOSIS_PROMPT,
                ],
                generation_config={
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json",
                },
            )
            try:
                text_block = response.text
            except Exception as text_exc:
                finish_reason = None
                try:
                    finish_reason = response.candidates[0].finish_reason
                except Exception:
                    pass
                raise ValueError(
                    f"No se pudo leer el texto de la respuesta (finish_reason={finish_reason}): {text_exc}"
                )
            if not text_block:
                raise ValueError("El modelo no devolvió texto")
            return extract_json(text_block)

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Intento %s falló: %s", attempt, exc)
            time.sleep(0.6 * attempt)

    raise last_error


def diagnosis_to_speech_text(diag):
    """Convierte el JSON del diagnóstico en un texto fluido para narrar,
    igual al que ya arma el frontend, pero generado en el backend para
    que /api/tts pueda usarse de forma independiente."""
    parts = []
    parts.append(f"Planta identificada: {diag.get('planta_identificada', 'no identificada')}.")
    parts.append(f"Problema detectado: {diag.get('plaga_o_problema', 'sin determinar')}.")
    if diag.get("severidad"):
        parts.append(f"Severidad: {diag['severidad']}.")
    if diag.get("urgencia"):
        parts.append(diag["urgencia"])

    symptoms = diag.get("sintomas_observados") or []
    if symptoms:
        parts.append("Síntomas observados:")
        for i, s in enumerate(symptoms, 1):
            parts.append(f"{i}. {s}.")

    steps = diag.get("pasos") or []
    if steps:
        parts.append("Plan de acción a seguir:")
        for i, p in enumerate(steps, 1):
            parts.append(f"Paso {i}: {p}.")

    if diag.get("prevencion"):
        parts.append(f"Prevención: {diag['prevencion']}.")

    return " ".join(parts)


def synthesize_speech(text):
    """Genera audio WAV (base64) a partir de texto usando Gemini TTS.
    Devuelve (base64_wav, error)."""
    if tts_client is None:
        return None, "El servidor no tiene configurada la clave de API."

    # Un tono cálido y claro, en español, apropiado para instrucciones a
    # agricultores. El prefijo "Di de forma clara y cálida en español:" ayuda
    # a que el modelo entienda que debe LEER el texto, no reaccionar a él.
    prompt = f"Di de forma clara, cálida y en español latino: {text}"

    try:
        response = tts_client.models.generate_content(
            model=TTS_MODEL_NAME,
            contents=prompt,
            config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": TTS_VOICE}
                    }
                },
            },
        )
        audio_data = response.candidates[0].content.parts[0].inline_data.data

        # El modelo devuelve PCM crudo a 24kHz/16-bit/mono; lo envolvemos en
        # un contenedor WAV para que cualquier reproductor lo entienda.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)

        wav_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return wav_b64, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo la síntesis de voz")
        return None, str(exc)


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "hoja-clinica-backend", "model": MODEL_NAME})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if model is None:
        return error_response(
            "El servidor no tiene configurada la clave de API de Gemini.",
            status=500,
            code="missing_api_key",
        )

    payload = request.get_json(silent=True)
    if not payload:
        return error_response("Falta el cuerpo JSON de la solicitud.")

    image_b64 = payload.get("image")
    media_type = payload.get("media_type", "image/jpeg")

    if not image_b64:
        return error_response("Falta el campo 'image' (base64).")

    if media_type not in ALLOWED_MIME_TYPES:
        return error_response(
            f"Tipo de imagen no soportado: {media_type}. Usa JPEG, PNG o WEBP.",
            code="unsupported_media_type",
        )

    approx_bytes = len(image_b64) * 3 / 4
    if approx_bytes > MAX_IMAGE_BYTES:
        return error_response(
            "La imagen es demasiado grande. Máximo 8 MB.",
            code="payload_too_large",
        )

    try:
        base64.b64decode(image_b64, validate=True)
    except Exception:
        return error_response("La imagen no es un base64 válido.", code="invalid_base64")

    try:
        diagnosis = call_gemini_with_retries(image_b64, media_type)
    except Exception:
        logger.exception("Fallo el análisis de imagen")
        return error_response(
            "No se pudo analizar la imagen en este momento. Intenta de nuevo.",
            status=502,
            code="analysis_failed",
        )

    required_keys = {"planta_identificada", "plaga_o_problema", "severidad", "pasos"}
    if not required_keys.issubset(diagnosis.keys()):
        return error_response(
            "El diagnóstico recibido está incompleto.", status=502, code="incomplete_diagnosis"
        )

    return jsonify({"ok": True, "diagnosis": diagnosis})


@app.route("/api/analyze-raw", methods=["POST"])
def analyze_raw():
    """Igual que /api/analyze pero acepta la imagen como bytes crudos en el
    cuerpo de la petición (pensado para Web1.PostFile de App Inventor, que no
    puede armar fácilmente un JSON con base64)."""
    if model is None:
        return error_response(
            "El servidor no tiene configurada la clave de API de Gemini.",
            status=500,
            code="missing_api_key",
        )

    image_bytes = request.get_data()
    if not image_bytes:
        return error_response("No se recibió ninguna imagen en el cuerpo de la solicitud.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return error_response(
            "La imagen es demasiado grande. Máximo 8 MB.",
            code="payload_too_large",
        )

    content_type = (request.content_type or "").split(";")[0].strip().lower()
    if content_type in ALLOWED_MIME_TYPES:
        media_type = content_type
    else:
        media_type = "image/jpeg"

    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    try:
        diagnosis = call_gemini_with_retries(image_b64, media_type)
    except Exception:
        logger.exception("Fallo el análisis de imagen (raw)")
        return error_response(
            "No se pudo analizar la imagen en este momento. Intenta de nuevo.",
            status=502,
            code="analysis_failed",
        )

    required_keys = {"planta_identificada", "plaga_o_problema", "severidad", "pasos"}
    if not required_keys.issubset(diagnosis.keys()):
        return error_response(
            "El diagnóstico recibido está incompleto.", status=502, code="incomplete_diagnosis"
        )

    return jsonify({"ok": True, "diagnosis": diagnosis})


@app.route("/api/tts", methods=["POST"])
def tts():
    """Convierte texto (o un diagnóstico completo) en audio con voz natural.
    Pensado para agricultores que no saben leer: reciben la misma
    información pero hablada, con una voz más natural que la síntesis de
    voz nativa del navegador.

    Body esperado, una de estas dos formas:
      {"text": "texto libre a narrar"}
      {"diagnosis": {...el objeto JSON del diagnóstico...}}
    """
    if tts_client is None:
        return error_response(
            "El servidor no tiene configurada la clave de API de Gemini.",
            status=500,
            code="missing_api_key",
        )

    payload = request.get_json(silent=True)
    if not payload:
        return error_response("Falta el cuerpo JSON de la solicitud.")

    if payload.get("diagnosis"):
        text = diagnosis_to_speech_text(payload["diagnosis"])
    else:
        text = (payload.get("text") or "").strip()

    if not text:
        return error_response("Falta el campo 'text' o 'diagnosis'.")

    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]

    audio_b64, err = synthesize_speech(text)
    if err:
        return error_response(
            "No se pudo generar el audio en este momento. Intenta de nuevo.",
            status=502,
            code="tts_failed",
        )

    return jsonify({"ok": True, "audio_base64": audio_b64, "mime_type": "audio/wav"})


@app.errorhandler(404)
def not_found(_):
    return error_response("Ruta no encontrada.", status=404, code="not_found")


@app.errorhandler(500)
def server_error(_):
    return error_response("Error interno del servidor.", status=500, code="server_error")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
