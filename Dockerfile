FROM python:3.10-slim

# Instala dependências do sistema para pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    && apt-get clean

# Define o diretório de trabalho
WORKDIR /app

# Copia o requirements e instala as dependências
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia todos os arquivos do projeto
COPY . .

# Expõe a porta usada pelo Dash
EXPOSE 8050

# Comando para rodar o Dash App
CMD ["python", "app.py"]
