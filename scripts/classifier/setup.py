import os

# Choose model
models = next(os.walk("models"))[1]
choose_model = "\nModel options:\n"
for n in range(len(models)):
    choose_model += f"[{n+1}] {models[n]}\n"
choose_model += f"\nChoose a model from above[1~{len(models)}]: "

model = models[int(input(choose_model)) - 1]

# Choose the model type
model_types = ["acc", "loss"]
choose_model_type = "\nModel type options:\n"
for n in range(len(model_types)):
    choose_model_type += f"[{n+1}] {model_types[n]}\n"
choose_model_type += f"\nChoose a model type from above[1~{len(models)}]: "

model_type = model_types[int(input(choose_model_type)) - 1]

# Choose params
params_sets = [p for p in os.listdir("params") if p[-4:] == ".txt"]
choose_params = "\nParameters set options:\n"
for n in range(len(params_sets)):
    choose_params += f"[{n+1}] {params_sets[n]}\n"
choose_params += f"\nChoose a set of parameters from above[1~{len(params_sets)}]: "

params = params_sets[int(input(choose_params)) - 1]

settings = {
    "model": model,
    "model_type": model_type,
    "params": params
}

with open("settings.txt", 'w') as f:
    f.write(str(settings))
print(settings)
