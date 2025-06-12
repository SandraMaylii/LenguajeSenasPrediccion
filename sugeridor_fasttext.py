import requests

def sugerir_palabra(letras, topn=5):
    letras = letras.strip().lower()
    if not letras:
        return []

    url = f"https://api.datamuse.com/words?sp={letras}*&v=es&max={topn}"
    response = requests.get(url)

    if response.status_code == 200:
        resultados = response.json()
        return [item['word'] for item in resultados]
    else:
        print("Error al conectarse a la API de Datamuse.")
        return []
