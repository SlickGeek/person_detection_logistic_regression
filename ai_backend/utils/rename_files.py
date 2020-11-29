import os


# Function to rename multiple files
def main():

    path = '/Users/davidbanda/Desktop/Object/'
    countStart = 808

    for count, filename in enumerate(os.listdir(path)):
        dst = "obj" + str(count+countStart) + ".jpg"
        src = path + filename
        dst = path + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)


if __name__ == '__main__':
    main()
