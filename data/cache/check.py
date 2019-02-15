import os
import cPickle as pickle
import pdb


def load_cache_file(filename):
    cache_file = os.path.join(filename)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = pickle.load(fid)
            return roidb
            
def save_cache_file(filename, roidb):
    with open(filename, 'w') as fid:
        for roi in roidb:
            pic_name = os.path.basename(roi['image'])
            pic_name = os.path.splitext(pic_name)[0]
            fid.write(pic_name+'\n')

def main():
    save = True
    f_path = 'voc_2007_train_gt_roidb.pkl'
    o_txt = 'train.txt'
    roidb = load_cache_file(f_path)
    if save:
        save_cache_file(o_txt,roidb)
    else:
        for roi in roidb:
            print roi
            pdb.set_trace()


if __name__ == '__main__':
    main()