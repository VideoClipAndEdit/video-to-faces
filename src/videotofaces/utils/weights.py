import os
import os.path as osp

import torch

from .download import url_download


def load_weights(model, link, filename=None, extra_conversion=None, sub=None, add_num_batches=False):
    """A common all-in-one function for loading pretrained weights from various sources into a model.
    Automatically downloads from ``link`` and saves is as "weights/``filename``.pt", then reads the
    weights dictionary (wd) from it and copies into the model's dictionary following the same order
    (using the fact that from Python 3.7 dictionaries are ordered). So the names in sources don't
    have to be the same, only the shapes and the order they appear.
    
    To ensure those conditions, a ref to some preprocessing func is passed to ``extra_conversion``
    (usually to reorder the source's weights or perform some other minor uninteresting alterations).

    Sources such as mmdetection also have weights under 'state_dict' key instead of at root (because
    there's usually a second key with meta params), so ``sub`` is used to "step into them" first.
    """
    fn = '%s.pt' % (filename or model.__class__.__name__.lower())
    wf = prep_file(link, fn)
    dv = torch.device(next(model.parameters()).device)
    wd_src = torch.load(wf, map_location=dv, weights_only=True)
    if sub:
        wd_src = wd_src[sub]
    if extra_conversion:
        wd_src = extra_conversion(wd_src)
    wd_dst = {}
    names = list(wd_src)
    shift = 0
    for i, w in enumerate(list(model.state_dict())):
        # some sources don't have 'num_batches_tracked' entries, but they only matter in train mode
        # and if BatchNorm2d 'momentum' param = None, so we can just insert them filled with 0
        if add_num_batches and w.endswith('num_batches_tracked'):
            #print('0 to ', w)
            wd_dst[w] = torch.tensor(0)
            shift += 1
        else:
            #print(names[i - shift], ' to ', w)
            wd_dst[w] = wd_src[names[i - shift]]
    model.load_state_dict(wd_dst)


def prep_file(link, fn):
    """An internal function used to download the file from ``link`` (either direct or Google Drive)
    and save it under ``weights`` folder with some log messages printed in the process.
    Called directly only from one-time conversion funcs (like migration code for onnx sources);
    every main pretrained model is calling ``load_weights`` instead.
    """
    gdrive = '://' not in link
    url = link if '://' in link else 'https://drive.google.com/uc?id=%s' % link
    if '__file__' in globals():
        # 4 levels above this code file, i.e. to the repo root
        home = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    else:
        home = os.getcwd()
    tdir = osp.join(home, 'weights')
    os.makedirs(tdir, exist_ok=True)
    dst = osp.join(tdir, fn)
    if osp.isfile(dst):
        print('Using weights from: ' + dst)
    else:
        print('Downloading weights from: ' + url)
        print('To: ' + dst)
        url_download(url, dst, gdrive)
    return dst