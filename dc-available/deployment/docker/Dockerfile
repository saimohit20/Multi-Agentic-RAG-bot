# Use more recent Python version with security patches
FROM --platform=linux/amd64 python:3.11.9-slim

# Install system dependencies with latest security patches
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    libsqlite3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install newer SQLite version (security hardened)
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3460000.tar.gz \
    && tar xvfz sqlite-autoconf-3460000.tar.gz \
    && cd sqlite-autoconf-3460000 \
    && ./configure --enable-fts5 --enable-rtree \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3460000* \
    && ldconfig

# Set secure environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV SQLITE_EXTENSIONS=/usr/local/lib/
ENV CHROMADB_FORCE_SQLITE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Verify SQLite version
RUN sqlite3 --version
RUN python -c "import sqlite3; print(sqlite3.sqlite_version)"

ARG PROJECT_DIR=aim205_actionable_llms
ARG WORK_DIR=/code
WORKDIR ${WORK_DIR}
ENV PYTHONPATH=/code

# Install security-patched Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies with fixed versions
RUN pip install --no-cache-dir pysqlite3-binary

# Install packages with CVE fixes
RUN pip install --no-cache-dir \
    "transformers>=4.44.0" \
    "langchain>=0.2.16" \
    "langchain-core>=0.2.38" \
    "langchain-openai>=0.1.23" \
    "langchain-community>=0.2.16" \
    "faiss-cpu>=1.8.0" \
    "openai>=1.45.0" \
    "chainlit>=1.3.2" \
    "streamlit>=1.30.0" \
    "pyautogen>=0.5.3,<0.6.0" \
    "comtypes>=1.4.0" \
    "pymupdf>=1.24.10" \
    "azure-identity>=1.17.1" \
    "scikit-image>=0.24.0" \
    "opencv-python-headless>=4.10.0" \
    "chardet>=5.2.0" \
    "extract-msg>=0.48.5" \
    "python-pptx>=1.0.0" \
    "pypdf>=4.3.1" \
    "streamlit-navigation-bar>=3.3.0" \
    "streamlit-extras>=0.4.7" \
    "pillow>=10.4.0" \
    "requests>=2.32.4" \
    "chromadb>=0.5.0" \
    "markdownify>=1.0.0" \
    "langdetect"

# Install PyTorch with latest security patches
RUN pip install --no-cache-dir torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies with security updates
RUN pip install --no-cache-dir \
    "sentence-transformers>=3.1.1" \
    "ipython>=8.27.0" \
    "ipykernel>=6.29.5" \
    "jupyter>=1.1.1" \
    "h11>=0.16.0" \
    "tornado>=6.5.0" \
    "jupyter_core>=5.8.1" \
    "setuptools>=78.1.1" \
    "pypdf2>=3.0.1" \
    "pip>=23.3" \
    "urllib3>=2.5.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create cache directories with proper permissions BEFORE switching user
RUN mkdir -p /tmp/autogen_cache && \
    chmod 777 /tmp/autogen_cache && \
    mkdir -p /home/appuser/.cache && \
    mkdir -p /home/appuser/.cache/huggingface && \
    mkdir -p /home/appuser/.cache/autogen && \
    mkdir -p /home/appuser/.diskcache && \
    chmod -R 777 /home/appuser/.cache && \
    chmod 777 /home/appuser/.diskcache

# Ensure /tmp has proper permissions for SQLite temporary files
RUN chmod 1777 /tmp

# Create application cache directory
RUN mkdir -p /code/.cache && \
    chmod 777 /code/.cache

# Copy project files with proper permissions
COPY --chown=appuser:appuser ./${PROJECT_DIR}/ /code/${PROJECT_DIR}/

# Copy additional main_data_test with proper permissions
COPY --chown=appuser:appuser ./${PROJECT_DIR}/../resources/DC_data/faiss_index /code/resources/DC_data/faiss_index
COPY --chown=appuser:appuser ./${PROJECT_DIR}/../resources/DC_data/images /code/resources/DC_data/images/
COPY --chown=appuser:appuser ./${PROJECT_DIR}/../resources/available_handbook /code/resources/available_handbook/
COPY --chown=appuser:appuser dc_available/common/img/team_members /code/dc_available/common/img/team_members

# Install Azure CLI with latest version
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Fix ownership of all directories for appuser AFTER copying files
RUN chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /code && \
    chown -R appuser:appuser /tmp/autogen_cache

# Set secure environment variables
ENV HF_HOME=/home/appuser/.cache/huggingface/
ENV AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1
ENV AZURE_CLI_SESSION_MODE=enabled
ENV AUTOGEN_USE_DOCKER=0
ENV AVAILABLE_APP_SOURCE="dc_available"
ENV IN_DOCKER=1
ENV AUTOGEN_CACHE_DISABLE=1

# Set cache-related environment variables for compatibility
ENV CACHE_DIR=/home/appuser/.cache
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
ENV AUTOGEN_CACHE_DIR=/tmp/autogen_cache
ENV DISKCACHE_DIR=/tmp/autogen_cache
ENV TMPDIR=/tmp
ENV TEMP=/tmp

# SQLite configuration for Docker compatibility
ENV SQLITE_TMPDIR=/tmp/autogen_cache
ENV SQLITE_ENABLE_FLOCK=0
ENV SQLITE_ENABLE_LOCKING_STYLE=0


# Expose the port for Streamlit
EXPOSE 8502

# Healthcheck for the container
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# Streamlit entrypoint
ENTRYPOINT ["streamlit", "run", "dc_available/frontend/Assistant.py", "--server.port=8502", "--server.address=0.0.0.0"]