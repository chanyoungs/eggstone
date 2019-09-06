import os

def run_setup():
    settings = {}
    
     # Choose model
    models = next(os.walk("models"))[1]
    choose_model = "\nModel options:\n"
    for n in range(len(models)):
        choose_model += f"[{n+1}] {models[n]}\n"
    choose_model += f"\nChoose a model from above[1~{len(models)}]: "

    settings["model"] = models[int(input(choose_model)) - 1]

    # Choose the model type
    model_types = ["acc", "loss"]
    choose_model_type = "\nModel type options:\n"
    for n in range(len(model_types)):
        choose_model_type += f"[{n+1}] {model_types[n]}\n"
    choose_model_type += f"\nChoose a model type from above[1~{len(models)}]: "

    settings["model_type"] = model_types[int(input(choose_model_type)) - 1]

    # Choose params
    params_sets = [p for p in os.listdir("params") if p[-4:] == ".txt"]
    choose_params = "\nParameters set options:\n"
    for n in range(len(params_sets)):
        choose_params += f"[{n+1}] {params_sets[n]}\n"
    choose_params += f"\nChoose a set of parameters from above[1~{len(params_sets)}]: "

    settings["params"] = params_sets[int(input(choose_params)) - 1]

    # Choose threshold
    while True:
        p = input("\nChoose probability threshold.\nThe higher the threshold, the higher the standard of quality and the more beans it will filter out.\nType a floating point number between 0~1[e.g. 0.5]: ")
        try:
            p = float(p)
            if 0 <= p and p <= 1:
                settings["p_threshold"] = p
                break
            else:
                print(f"\nPlease type a number between 0 and 1. (You typed '{p}'): ")
        except ValueError:
            print(f"\nPlease type a number between 0 and 1. (You typed '{p}'): ")

    with open("settings.txt", 'w') as f:
        f.write(str(settings))
    print(settings)

if __name__ == '__main__':
    run_setup()
