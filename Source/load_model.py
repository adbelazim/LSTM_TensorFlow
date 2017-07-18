from models import model_from_json

#model.set_weights(weights)
#model.get_weights()
#model.to_json()
#model.save_weights(filepath)
#model.load_weights(filepath, by_name=False)

##TO DO

#acces to path with model 11,55,99,19,91
model = model_from_json("/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/Stateless/10_time_steps/19/2_layer/8_units/1_fold")
#retornar modelo de cada caso
#a cada modelo obtenerle los pesos
#cargar los pesos segun sea el caso (esto se debe hacer en run network)