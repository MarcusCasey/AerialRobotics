
#marcus casey
#anurag kulkarni
#clear vision library
#ir_hog_svm


#import libraries
import cv2
import sys
import os

#grab images from image directory
def locateFile(path, filter=''):
    #file_list is for storing all files in a directory into an []
    file_list = []
    #grab all files in directory, name doesn't matter, filter applies relevant directory listing semantics
    for (root, dirs, files) in os.walk(path):
        if path == root:
            for file in files:
                if file.find(filter) > -1:
                    file_list.append(os.path.join(root,file).replace("\\", "/"))
    return file_list


#main fnc of hog svm
def ir_hog_svm(img_location, show=False):
    #load image from path
    mainImage = cv2.imread(img_location)
    #grab shape of image
    #numpy index the shape
    main_height, main_width = mainImage.shape[:2]
    #resultant size
    size = (512, int(512 * main_height / main_width))
    #resize image to new size
    mainImage = cv2.resize(mainImage, size)
    #create hog descriptor and detector with default params
    hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
    #set coefficients for linear SVM classifier to the hogdescriptor of coefficients for classifier traned people detection
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #performs into a multi scale window
    locations, r = hog.detectMultiScale(mainImage, winStride=(8, 8), padding=(32, 32), scale=1.05, hitThreshold=0, finalThreshold=1)
    #apply
    for (x, y, w, h) in locations:
        cv2.rectangle(mainImage, (x, y),(x+w, y+h),(255,255,0), 3)
            #display image
    if show:
        cv2.imshow("detect image",mainImage)
        cv2.waitKey(0)
#return image
    return mainImage

#correct directory call
if __name__ == '__main__':
    #hold directory name
    argv = sys.argv
    argc = len(argv)
    #if no directory entered
    if (argc != 2):
        print ('Define path name after run' %argv[0])
        quit()
    # img directory = held directory name
    image_dir = argv[1]
#  grab png's ONLY from directory
    image_paths = locateFile(image_dir, '.png')
# result directory name
    applied_directory = './applied'
# create result directory if none exists
    if os.path.exists(applied_directory) == False:
        os.makedirs(applied_directory)
#apply ir_hog_svm detection images in path
    for image_path in image_paths:
        #apply to directory
        mainImage = ir_hog_svm(image_path)
        #filename joined with directory on completion and successful grab
        output_file = applied_directory + "/result_" + os.path.basename(image_path)
        #save final file to new directory
        cv2.imwrite(output_file,mainImage)
        print("applied to:", image_path)
