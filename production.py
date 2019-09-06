import sys
import os
import shutil

name = input("Type the production name: ")

# Remove Previous production folder
if os.path.isdir(name):
    remove = input(f"Production '{name}' already exists.\nReplace the existing production?[y/n]: ")
    if remove != "y":
        sys.exit()
    shutil.rmtree(name)
    print(f"\nExisting production '{name}' removed.")

# Create production folder
os.makedirs(name)
print(f"\nNew folder '{name}' created.")

# Copy preprocessor.py
shutil.copy2(os.path.join("scripts", "preprocessor", "preprocessor.py"), os.path.join(name, "preprocessor.py"))
print("\npreprocessor.py copied")

# Copy params
shutil.copytree(os.path.join("scripts", "preprocessor", "params"), os.path.join(name, "params"))
print(f"\nPreprocessing parameters copied")

# Copy classifier.py
shutil.copy2(os.path.join("scripts", "classifier", "classifier.py"), os.path.join(name, "classifier.py"))
print("\nclassifier.py copied")

# Copy setup.py
shutil.copy2(os.path.join("scripts", "classifier", "setup.py"), os.path.join(name, "setup.py"))
print("\nsetup.py copied")

# Copy settings.txt
shutil.copy2(os.path.join("scripts", "classifier", "settings.txt"), os.path.join(name, "settings.txt"))
print("\nsettings.txt copied")

# Copy demo.py
shutil.copy2(os.path.join("scripts", "classifier", "demo.py"), os.path.join(name, "demo.py"))
print("\ndemo.py copied")

# Copy data_demo
shutil.copytree(os.path.join("scripts", "classifier", "data_demo"), os.path.join(name, "data_demo"))
print(f"\nPreprocessing data_demo copied")

# Copy all models
model_folders = next(os.walk("outputs"))[1]
for model in model_folders:
    shutil.copytree(os.path.join("outputs", model, "checkpoints"), os.path.join(name, "models", model))
print("\nModels copied")
