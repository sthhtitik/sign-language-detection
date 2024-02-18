import os

for file_path in os.listdir("data"):
    for files in os.listdir(os.path.join("data", file_path)):
        for i in range(100):
            if os.path.exists(os.path.join("data", file_path, f"{i}.jpg")):
                os.remove(os.path.join("data", file_path, f"{i}.jpg"))
