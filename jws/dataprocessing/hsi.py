
import scipy.io as sio
import os

def load():
    DATASET_DIR = '/media/data/Datasets/hyperspectral-images/'
    
    ims = []
    gts = []

    pairs = []

    for fname in os.listdir(DATASET_DIR):
        if fname.endswith('_gt.mat'):
            gts.append(DATASET_DIR + fname)
        elif fname.endswith('.mat'):
            ims.append(DATASET_DIR + fname)
            
    print(gts)
            
    ims.sort()
    gts_match = []
    for im in ims:
        for gt in gts:
            if 'Pavia' in im: print(im[:-4] + ' ' +  gt[:-7], im[:-4] == gt[:-7])
            if im[:-4] == gt[:-7]:
                gts_match.append(gt)
                break
        
            
    gts = gts_match
    pairs = [{'image': fname,'labels': gt} for fname, gt in zip(ims, gts)]
    for p in pairs: print(p)
    
    ds = []
        
    for p in pairs:
        im_mat = sio.matlab.loadmat(p['image'])
        labels_mat = sio.matlab.loadmat(p['labels'])
        im_name = list(im_mat.keys())[-1]
        label_name = list(labels_mat.keys())[-1]
        print(im_name, im_mat[im_name].shape, labels_mat[label_name].shape)
        ds.append(
            {
                'name': im_name,
                'image': im_mat[im_name],
                'labels': labels_mat[label_name]
            }
        )
        
    return ds

if __name__ == '__main__':
    load()