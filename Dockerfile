FROM python:3.10-slim

WORKDIR /app

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Crear un usuario no privilegiado para ejecutar la aplicación
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Exponer el puerto que se usará (aunque se define dinámicamente con PORT)
EXPOSE ${PORT}

# Ejecutar la aplicación con waitress
CMD waitress-serve --host=0.0.0.0 --port=${PORT} app:app
