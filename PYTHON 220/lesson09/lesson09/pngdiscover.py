def png_file():
    """function that returns png files"""
    file_list = list()
    for i in range(0,100):
        file_list.append("file" + str(i) + ".png")
    return file_list

if __name__ == "__main__":
    FILE_LIST = png_file()
