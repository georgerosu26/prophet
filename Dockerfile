FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_DIR=/app/models \
    CMDSTAN=/opt/cmdstan/cmdstan-2.38.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Prophet requires a Stan backend; install CmdStan for cmdstanpy.
RUN python -c "from cmdstanpy import install_cmdstan; install_cmdstan(dir='/opt', cores=2)"

# Prophet 1.1.x may force a packaged cmdstan path (cmdstan-2.33.1).
# Replace that bundled path with a symlink to the valid installed CmdStan.
RUN python - <<'PY'\nimport os, site\nsite_pkg = site.getsitepackages()[0]\nprophet_stan = os.path.join(site_pkg, 'prophet', 'stan_model')\nos.makedirs(prophet_stan, exist_ok=True)\ntarget = '/opt/cmdstan/cmdstan-2.38.0'\nlink = os.path.join(prophet_stan, 'cmdstan-2.33.1')\nif os.path.islink(link) or os.path.exists(link):\n    try:\n        if os.path.islink(link):\n            os.unlink(link)\n        elif os.path.isdir(link):\n            import shutil\n            shutil.rmtree(link)\n        else:\n            os.remove(link)\n    except Exception:\n        pass\nos.symlink(target, link)\nprint('Linked', link, '->', target)\nPY

COPY app /app/app

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
