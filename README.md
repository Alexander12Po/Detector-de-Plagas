# Hoja Clínica — Diagnóstico de plagas en plantas

App con tres pasos:
1. El usuario sube o toma una foto de su planta.
2. Un backend en Python llama a la IA (Claude) para diagnosticar la plaga/enfermedad y dar un plan de tratamiento.
3. Se localizan viveros, tiendas de jardinería y agrónomos cercanos usando la geolocalización del navegador (gratis, sin API de mapas de pago).

```
planta-diagnostico/
├── backend/
│   ├── app.py            servidor Flask con el endpoint /api/analyze
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
└── README.md
```

## 1. Backend (Python)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y coloca tu clave de la API de Anthropic:

```
ANTHROPIC_API_KEY=sk-ant-tu-clave-aqui
```

Consigue tu clave gratis (con crédito inicial de prueba) en https://console.anthropic.com/

Levanta el servidor:

```bash
python app.py
```

El backend queda corriendo en `http://localhost:5000`.
Puedes probarlo con: `curl http://localhost:5000/api/health`

## 2. Frontend

**Para probarlo rápido:** abre `frontend_completo.html` directamente (doble clic o
desde el celular). Ese archivo trae el CSS y el JavaScript ya integrados, así que
funciona solo, sin depender de otros archivos.

**Para producción/despliegue:** usa la carpeta `frontend/` (con `index.html`,
`styles.css` y `script.js` separados). Sirve igual, pero organizado en archivos
distintos — ideal para subir a Netlify, Vercel, etc. También puedes servirla
localmente con:

```bash
cd frontend
python -m http.server 8080
```

y visita `http://localhost:8080`.

Si tu backend corre en otra URL (por ejemplo cuando lo despliegues en internet),
agrega esta línea antes de `script.js` en `index.html`:

```html
<script>window.HOJA_CLINICA_API_URL = "https://tu-backend-desplegado.com";</script>
<script src="script.js"></script>
```

## 3. Despliegue gratuito recomendado

- **Backend**: Render.com o Railway.app (plan gratuito). Sube la carpeta `backend/`,
  configura la variable de entorno `ANTHROPIC_API_KEY` en su panel, y listo.
- **Frontend**: Netlify, Vercel o GitHub Pages (gratis). Sube la carpeta `frontend/`.

## Notas de seguridad

- La clave de la API **nunca** se coloca en el frontend ni en el navegador: vive solo
  en el backend, en la variable de entorno `ANTHROPIC_API_KEY`. Así nadie puede robarla
  inspeccionando el código de la página.
- El backend valida tipo y tamaño de imagen, reintenta automáticamente si la IA falla
  una vez, y siempre responde con un mensaje de error claro en vez de "caerse".

## Costos

- El uso del backend, el frontend y el hosting recomendado es gratuito.
- La API de Anthropic no es gratuita de forma indefinida, pero incluye crédito de
  prueba y su costo por análisis de imagen es muy bajo (fracciones de centavo por foto).
