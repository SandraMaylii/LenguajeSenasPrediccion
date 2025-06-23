from imutils.video import VideoStream
from flask import Response, request, Flask, render_template, jsonify
import threading
import argparse
import time
import cv2
import numpy as np
import torch
from model import Net
from sugeridor_fasttext import sugerir_palabra

# Cargar modelo de PyTorch
model = torch.load('model_trained.pt')
model.eval()

# Diccionario de clases
signs = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'
}

# Variables globales
outputFrame = None
lock = threading.Lock()
trigger_flag = False

buffer_letras = ""
palabras_sugeridas = []
palabra_elegida = ""
ultima_letra = ""
ultimo_tiempo_letra = time.time()

app = Flask(__name__)
vc = VideoStream(src=0).start()
time.sleep(2.0)

def detect_gesture(frameCount):
    global vc, outputFrame, lock
    global buffer_letras, palabras_sugeridas, ultima_letra, ultimo_tiempo_letra

    while True:
        frame = vc.read()
        frame = cv2.resize(frame, (700, 480))
        frame = cv2.flip(frame, 1)  # ← CORRIGE efecto espejo horizontal

        img = frame[20:250, 20:250]

        res = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        res1 = np.reshape(res, (1, 1, 28, 28)) / 255
        res1 = torch.from_numpy(res1).type(torch.FloatTensor)

        out = model(res1)
        probs, label = torch.topk(out, 25)
        probs = torch.nn.functional.softmax(probs, 1)
        pred = out.max(1, keepdim=True)[1]

        if float(probs[0, 0]) >= 0.97 and (time.time() - ultimo_tiempo_letra) >= 2:
            letra = signs[str(int(pred))].lower()
            if letra != ultima_letra:
                buffer_letras += letra
                ultima_letra = letra
                ultimo_tiempo_letra = time.time()

                sugerencias = sugerir_palabra(buffer_letras, topn=8)
                palabras_sugeridas.clear()
                palabras_sugeridas.extend(sugerencias)

        detected = 'No se detecta nada' if float(probs[0, 0]) < 0.4 else signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0, 0])) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, detected, (60, 285), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

        with lock:
            outputFrame = frame.copy()

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

# === FLASK ROUTES ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/trigger") 
def trigger():
    global trigger_flag
    trigger_flag = True
    return Response('done')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/texto_actual")
def texto_actual():
    global buffer_letras
    return jsonify(buffer_letras)

@app.route("/sugerencias")
def get_sugerencias():
    return jsonify(palabras_sugeridas)

@app.route("/seleccionar")
def seleccionar():
    global palabra_elegida, buffer_letras, ultima_letra
    palabra = request.args.get("palabra", "")
    if palabra:
        palabra_elegida += palabra + " "
    buffer_letras = ""
    ultima_letra = ""
    return Response("ok")

@app.route("/palabra_seleccionada")
def palabra_seleccionada():
    return jsonify(palabra_elegida)

@app.route("/borrar_ultima")
def borrar_ultima():
    global buffer_letras, palabras_sugeridas
    if buffer_letras:
        buffer_letras = buffer_letras[:-1]
        palabras_sugeridas = sugerir_palabra(buffer_letras, topn=8) if buffer_letras else []
    return Response("ok")

@app.route("/borrar_texto_completo")
def borrar_texto_completo():
    global buffer_letras, palabras_sugeridas
    buffer_letras = ""
    palabras_sugeridas = []
    return Response("ok")

@app.route("/borrar_ultima_seleccionada")
def borrar_ultima_seleccionada():
    global palabra_elegida
    palabras = palabra_elegida.strip().split()
    if palabras:
        palabras.pop()
    palabra_elegida = ' '.join(palabras) + (" " if palabras else "")
    return Response("ok")


@app.route("/borrar_seleccionadas")
def borrar_seleccionadas():
    global palabra_elegida
    palabra_elegida = ""
    return Response("ok")

@app.route("/limpiar")
def limpiar():
    global buffer_letras, palabras_sugeridas, palabra_elegida, ultima_letra
    buffer_letras = ""
    palabras_sugeridas.clear()
    palabra_elegida = ""
    ultima_letra = ""
    return Response("ok")

@app.route("/estado")
def estado():
    return jsonify({
        "texto": buffer_letras,
        "sugerencias": palabras_sugeridas,
        "seleccionada": palabra_elegida
    })

# === MAIN ===

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="IP address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="Port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32, help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_gesture, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

vc.stop()
