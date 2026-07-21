FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

COPY . .

# .system/ (nexus.shared) is expected to be bind-mounted or copied in
# before the container starts - install.py fails fast with a clear message
# if it's missing or version-incompatible instead of a raw ImportError.
CMD ["sh", "-c", "python install.py && python run.py"]
