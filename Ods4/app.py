from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = pickle.load(open('modelo_arbol.pkl', 'rb'))

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        datos = [
            int(request.form['motivado']),
            int(request.form['abandono']),
            int(request.form['inasistencias']),
            int(request.form['trabaja']),
            int(request.form['apoyo_familiar']),
            int(request.form['estres']),
            int(request.form['confianza'])
        ]
        
        entrada = np.array([datos])
        prediccion = modelo.predict(entrada)[0]

        if prediccion == 1:
            resultado = "ALTO RIESGO DE DESERCIÓN"
        else:
            resultado = "RIESGO BAJO / CONTROLADO"

        return f"<h2 class='text-center text-warning mt-5'>{resultado}</h2>"
    
    except Exception as e:
        return f"<h4 class='text-danger'>Error: {e}</h4>"

if __name__ == '__main__':
    app.run(debug=True)
