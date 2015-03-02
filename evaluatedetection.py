'''
Train and test a classifier for detecting objects in microscopy images. The
detection breaks each image down into a set of overlapping patches using a 
sliding window. Patches are labelled as positive or negative according to
whether they overlap with the bounding box of an object of interest. For each 
patch, a set of morphological features are calculated, which allows 
classification to be carried out.
'''

import pylab as pl
import cv2
import shapefeatures
import os
from lxml import etree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn import ensemble
import numpy as np
import glob
import pickle

DATA_DIR = 'data/'
IMAGE_DIR = DATA_DIR + 'images/'
FEATURES_DIR = DATA_DIR + 'features/'
ANNOTATION_DIR = DATA_DIR + 'annotation/'
RESULTS_DIR = DATA_DIR + 'results/'


def get_bounding_boxes_for_single_image(annofilename):
    '''
    Given an annotation XML filename, get a list of the bounding boxes around
    each object (the ground truth object locations).
    '''
    annofileexists = os.path.exists(annofilename)
    boundingboxes = []

    if (annofileexists):
        # Read the bounding boxes from xml annotation
        tree = etree.parse(annofilename)
        r = tree.xpath('//bndbox')
        
        bad = tree.xpath('//status/bad')
        badimage = (bad[0].text=='1')
        
        if badimage: 
            print 'Bad image: ' + annofilename
            exit

        if (len(r) != 0):
            for i in range(len(r)):
                xmin = round(float(r[i].xpath('xmin')[0].text))
                xmin = max(xmin,1)
                xmax = round(float(r[i].xpath('xmax')[0].text))
                ymin = round(float(r[i].xpath('ymin')[0].text))
                ymin = max(ymin,1)
                ymax = round(float(r[i].xpath('ymax')[0].text))
                xmin, xmax, ymin, ymax = int(xmin),int(xmax),int(ymin),int(ymax)

                boundingboxes.append((xmin,xmax,ymin,ymax))
                    
    return boundingboxes
    
    
def get_patch_labels_for_single_image(imgfilename, width, height, size, step):
    '''
    Read the XML annotation files to get the labels of each patch for a 
    given image. The labels are 0 if there is no object in the corresponding
    patch, and 1 if an object is present.
    '''
    annotationfilename = ANNOTATION_DIR + imgfilename[:-3] + 'xml'
    boundingboxes = get_bounding_boxes_for_single_image(annotationfilename)

    # Scan through patch locations in the image
    labels = []
    x = step
    y = step
    while y<height:
        x = step;
        while (x<width):
            objecthere=0
            for bb in boundingboxes:
                margin = 0
                xmin = bb[0] + margin
                xmax = bb[1] - margin
                ymin = bb[2] + margin
                ymax = bb[3] - margin
                
                left = x - step/2
                right = x + step/2
                top = y - step/2
                bottom = y + step/2
                
                xbb = (xmin+xmax)/2
                ybb = (ymin+ymax)/2

                if (x>xmin and x<xmax and y>ymin and y<ymax):
                    objecthere = 1
                    break

            # Output the details for this patch
            labels.append(objecthere)

            x+=step
        y += step
    
    return np.array(labels)


def getfeatures(baseimgfilenames, size, step,
                attributes=None,
                filters=None, 
                centiles=[10,30,50,70,90],
                loadfromfile=False, 
                savetofile=False, 
                filename='features.npy'): 
    '''
    Calculate feature vectors for every patch in a set of images.
    '''
             
    filename = FEATURES_DIR + filename 
    if loadfromfile and os.path.isfile(filename):
        features = np.load(filename)
    else:
        instance_list = []

        for image_idx in range(len(baseimgfilenames)):
        # Read in the data for many files
            basefilename = baseimgfilenames[image_idx]
            print '%s (%d/%d)' % (basefilename, image_idx+1,
                                  len(baseimgfilenames))
            img = cv2.imread(IMAGE_DIR + basefilename, cv2.IMREAD_GRAYSCALE)
            curr_instances = shapefeatures.patchextract(img, size, step, 
                                                        attributes=attributes,
                                                        filters=filters, 
                                                        centiles=centiles,
                                                        momentfeatures=True)
            instance_list.append(curr_instances)
      
        features = np.vstack(instance_list)
                                             
    if savetofile and not loadfromfile:
        np.save(filename, features)
        
    return features


def getlabels(baseimgfilenames, size, step,
              loadfromfile=False, savetofile=False,
              filename='labels.npy'):
    '''
    Calculate labels for every patch in a set of images.
    '''
    filename = FEATURES_DIR + filename 
    if loadfromfile and os.path.isfile(filename):
        labels = np.load(filename)  
    else:    
        labels_list = []
        
        for image_idx in range(len(baseimgfilenames)):
        # Read in the data for many files
            basefilename = baseimgfilenames[image_idx]
            #print '%s (%d/%d)' % (basefilename, image_idx+1, num_files)
                                                              
            # Calculate labels for each patch (look up XML)
            img = cv2.imread(IMAGE_DIR + basefilename, cv2.IMREAD_GRAYSCALE)
            curr_labels = get_patch_labels_for_single_image(basefilename,
                                                            img.shape[1],
                                                            img.shape[0],
                                                            size, step)

            labels_list.append(curr_labels)
      
        labels = np.hstack(labels_list)        
    
    if savetofile and not loadfromfile:
        np.save(filename, labels)  
        
    return labels
    
      
def displayframe(img, predictions, labels, threshold, size, step):
    '''
    Show a single image with colour coded boxes indicating true positives,
    false positives and false negatives.
    '''
    # Generate the coordinates of each region in the sequence
    linethickness = 2
    idx = 0
    y = step
    height = img.shape[0]
    width = img.shape[1]
    while y<height:
        x = step;
        while (x<width):
            left = x-(size/2)
            right = x+(size/2)
            top = y-(size/2)
            bottom = y+(size/2)
            
            if predictions[idx]>threshold and labels[idx]:
                # TP: draw a blue box
                cv2.rectangle(img,(left,top),(right,bottom), [255, 0, 0],
                              linethickness)
                              
            elif predictions[idx]>threshold and not labels[idx]:
                # FP: draw a red box
                cv2.rectangle(img,(left,top),(right,bottom), [0, 0, 255],
                              linethickness)
                              
            elif labels[idx]:
                # FN: draw a white circle
                cv2.rectangle(img,(left,top),(right,bottom), [255, 255, 255],
                              linethickness)                
            
            idx += 1
            x += step
        y += step
    
    cv2.imshow('result',img)
    cv2.waitKey(0)
    return None
    
if __name__=='__main__':
    '''
    Available morphological features
    --------------------------------
    0: Area
    1: Area of min. enclosing rectangle
    2: Square of diagonal of min. enclosing rectangle
    3: Cityblock perimeter
    4: Cityblock complexity (Perimeter/Area)
    5: Cityblock simplicity (Area/Perimeter)
    6: Cityblock compactness (Perimeter^2/(4*PI*Area))
    7: Large perimeter
    8: Large compactness (Perimeter^2/(4*PI*Area))
    9: Small perimeter
    10: Small compactness (Perimeter^2/(4*PI*Area))
    11: Moment of Inertia
    12: Elongation: (Moment of Inertia) / (area)^2
    13: Mean X position
    14: Mean Y position
    15: Jaggedness: Area*Perimeter^2/(8*PI^2*Inertia)
    16: Entropy
    17: Lambda-max (Max.child gray level - current gray level)
    18: Gray level
    '''

    featureset = [3,7,11,12,15,17]
    num_files = 2703
    train_set_proportion = .8
    test_set_proportion = 1 - train_set_proportion
    filters = [[11,'>',1000]]
    centiles = [0,25,50,75,100]
    size = 40
    step = 30
    reusefeatures = False
    savefeatures = True
    reuseclassifier = False
    saveclassifier = False
    saveresults = False
                            
    # Split up image files into training and test sets
    imgfilenames = glob.glob(IMAGE_DIR + '*.jpg')
        
    baseimgfilenames = [os.path.basename(imgfilenames[i]) 
                        for i in range(num_files)]                            
                                                   
    train, test = train_test_split(np.arange(num_files),
                                   train_size=train_set_proportion,
                                   test_size=test_set_proportion,
                                   random_state=1)  
                                   
    trainfiles = [baseimgfilenames[i] for i in train]
    testfiles = [baseimgfilenames[i] for i in test]
                          

    print 'Extracting features from training images...'
    Xtrain = getfeatures(trainfiles, size, step,
                        attributes=featureset,
                        filters=filters,
                        centiles=centiles,
                        loadfromfile=reusefeatures, 
                        savetofile=savefeatures,
                        filename='Xtest.npy') 
                                                                                 
    ytrain = getlabels(trainfiles, size, step,
                      loadfromfile=reusefeatures,
                      savetofile=savefeatures,
                      filename='ytest.npy')     
                            
                         
    
    print 'Extracting features from testing images...'
    Xtest = getfeatures(testfiles, size, step,
                        attributes=featureset,
                        filters=filters,
                        centiles=centiles,
                        loadfromfile=reusefeatures, 
                        savetofile=savefeatures,
                        filename='Xtest.npy') 
                                                                                 
    ytest = getlabels(testfiles, size, step,
                      loadfromfile=reusefeatures,
                      savetofile=savefeatures,
                      filename='ytest.npy')     
                            
    if reuseclassifier:
        classifier = pickle.load(open(FEATURES_DIR + 'classifier.pkl', 'rb')) 
        
    else:
        print 'Training...'
        classifier = ensemble.ExtraTreesClassifier(n_estimators=500, 
                                               max_depth=10, n_jobs=-1)
        classifier.fit(Xtrain,ytrain)
        
    if saveclassifier:
        pickle.dump(classifier, open(FEATURES_DIR + 'classifier.pkl', 'wb'))
                           
    
    print 'Testing...'
    predictions = classifier.predict_proba(Xtest)[:,1]

    # Assume that examples with all-zero features (no shapes in patch) are negative    
    negativetest = Xtest[:,0]==0
    predictions[negativetest]=0
    ytest[negativetest]=0

    # Calculate the performance
    precision, recall, thresholds = precision_recall_curve(
                                        ytest, predictions)
    
    area = auc(recall, precision)
    print("Area under precision-recall curve: %0.2f" % area)
    
    fig = pl.figure()
    fig.set_size_inches(4,4)
    pl.plot(recall, precision)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.grid(True)
    pl.ylim([0.0, 1.0])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall: AUC=%0.2f' % area)
    
    if saveresults:
        np.save('data/results/predictions.npy', predictions)
        pickle.dump(classifier, open(RESULTS_DIR + 'classifier.pkl', 'wb'))
