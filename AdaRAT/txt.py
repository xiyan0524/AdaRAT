def clear(path):
    with open(path, "a") as f:
        f.seek(0)
        f.truncate()
        f.close()

def write(path, txt):
    with open(path, "a") as f:
        f.write(txt)
        f.write("\n")
        f.close()
