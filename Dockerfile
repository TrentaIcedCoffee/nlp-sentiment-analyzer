FROM --platform=linux/amd64 python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENV debug=false

CMD ["sh", "-c", "python main.py --debug=${debug}"]