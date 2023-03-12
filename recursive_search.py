# Import the os module, for the os.walk function
import os

fileName = []
directName = []
# Set the directory you want to start from
rootDir = "C:\\Users\\briank\\Documents\\coreqa\\python\\openCV"
fhandle = open("file_locations.csv", "w")


def fileSearch(starts="", ends=""):
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            # Set the file names you are looking for use startswith for name of file and endswith for file type
            if fname.lower().startswith((starts)) & fname.lower().endswith((ends)):
                print("%s" % fname)
                # Sticking the directory name and file name into lists so they are callable later and writing to the Output.csv
                directName.append(dirName)
                fileName.append(fname)
                fhandle.write(dirName + "\\" + fname + "\n")


# Here an simple example usage that finds files that start with cat and are in the format jpeg
if __name__ == "__main__":
    fileSearch("", "jpg")
    fhandle.close()
