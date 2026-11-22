# humanorbot web page

A simple React (Vite) page: paste a text, choose the base or the fine-tuned model, and see how likely the text is to be AI-generated.

Live: https://abdullahahmad0629-glitch.github.io/humanorbot/

The page calls the model server (`src/app.py`), which runs on Kaggle via `kaggle/api_server.py`. Paste the server's address into **Model server address**; it is remembered in your browser. A link with `?api=<address>` fills it in.

GitHub Pages rebuilds the page automatically (`.github/workflows/deploy.yml`) whenever something in `web/` changes.

## Run locally

```
cd web
npm install
npm run dev
```
