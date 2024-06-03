from fastapi import FastAPI
import torch
import joblib
import pickle
import io

# Método para cargar el modelo y el tokenizador desde el archivo .sav
"""def load_model_and_tokenizer(model_path):
    # Cargar el modelo y el tokenizador desde el archivo .sav
    model_tokenizer_dict = joblib.load(model_path)
    loaded_model = model_tokenizer_dict['model']
    loaded_tokenizer = model_tokenizer_dict['tokenizer']
    # Asegurar que el modelo se cargue en la CPU
    loaded_model = loaded_model.to(torch.device('cpu'))
    return loaded_model, loaded_tokenizer"""

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_model_and_tokenizer(model_path):
    try:
        # Cargar el modelo y el tokenizador desde el archivo .sav utilizando CPU_Unpickler
        with open(model_path, 'rb') as f:
            model_tokenizer_dict = CPU_Unpickler(f).load()
        loaded_model = model_tokenizer_dict['modelo']
        loaded_tokenizer = model_tokenizer_dict['tokenizador']
    except Exception as e:
        raise RuntimeError(f"Error loading model and tokenizer: {str(e)}")
    return loaded_model, loaded_tokenizer

app = FastAPI()

@app.get('/')
def hello():
    return {'message':'Hello World'}

@app.post('/predict') 
def predict(request: dict):
    text = request['text']  # Asumo que el texto de entrada se envía en la clave 'text' del request
    #model_path = "Models/modelo_y_tokenizador.sav"
    model_path = "Models/modelo_bert.pkl"
    # Carga el modelo y el tokenizer utilizando la función existente
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Preprocesamiento del texto -
    encoded_text = tokenizer(text, return_tensors="pt")

    # Desactiva el gradiente para mejorar la eficiencia en inferencia
    with torch.no_grad():
        outputs = model(**encoded_text)
        predicted_class = torch.argmax(outputs.logits).item()

    return {'prediction': predicted_class}