'''
Show examples of image patches classified arranged by different classification
probabilities. After running the classification experiment, this file can be 
run by sourcing in the same ipython environment:
run -i plot-patches-by-classification-score.py
'''


thresholds = [.6, .3, 0]
ncols = 8
nrows = len(thresholds)
for row in range(nrows):
    thresh = thresholds[row]
    idx = np.where(np.logical_and(predictions>thresh, predictions<(thresh+.1)))[0]
    idx = np.random.permutation(idx)
    npatchesperimage = len(ytest)/len(testfiles)
    offset = 0
    for i in range(ncols):
        found = False
        while not found:
            imageidx = idx[i+offset]/npatchesperimage
            patchidx = idx[i+offset] % npatchesperimage
            if patchidx<300:
                found = True
            else:
                offset += 1
        
        fname = IMAGE_DIR + testfiles[imageidx]
        img = cv2.imread(fname)
        height, width, channels = img.shape
        patchcount = 0
        finished = False
        x = step
        y = step
        while y<height and not finished:
            x = step;
            while (x<width) and not finished:
                if patchcount==patchidx:
                    left = x - size/2
                    right = x + size/2
                    top = y - size/2
                    bottom = y + size/2 
                    patch = img[top:bottom, left:right, :]
                    finished = True
                patchcount += 1
                x+=step
            y += step
        
        plt.subplot(nrows, ncols, row*ncols + i)
        plt.imshow(patch)
        plt.xticks([])
        plt.yticks([])
        if i==1:
            ylabel('%.1f-%.1f' % (thresh, thresh+.1))
    

plt.savefig('output-patches-by-threshold.pdf', bbox_inches='tight')
