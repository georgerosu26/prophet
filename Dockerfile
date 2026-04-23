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
RUN python -c "import os,site,shutil;site_pkg=site.getsitepackages()[0];prophet_stan=os.path.join(site_pkg,'prophet','stan_model');os.makedirs(prophet_stan,exist_ok=True);target='/opt/cmdstan/cmdstan-2.38.0';link=os.path.join(prophet_stan,'cmdstan-2.33.1');(os.unlink(link) if os.path.islink(link) else (shutil.rmtree(link) if os.path.isdir(link) else (os.remove(link) if os.path.exists(link) else None)));os.symlink(target,link);print('Linked',link,'->',target)"

COPY app /app/app

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
