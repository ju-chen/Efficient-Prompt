import os
import cv2
import torch
import numpy as np 
from PIL import Image
from clip import clip
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop



def load_clip_cpu(backbone_name):
    model_path = 'path_to_CLIP_ViT-B-16_pre-trained_parameters'  
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    model = clip.build_model(state_dict or model.state_dict())

    return model


def transform_center():
    interp_mode = Image.BICUBIC
    tfm_test = []
    tfm_test += [Resize(224, interpolation=interp_mode)] 
    tfm_test += [CenterCrop((224,224))]
    tfm_test += [ToTensor()]
    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    tfm_test += [normalize]
    tfm_test = Compose(tfm_test)

    return tfm_test


def get_videos(vidname, read_path):
    allframes = []
    videoins = read_path + vidname
    vvv = cv2.VideoCapture(videoins)
    if not vvv.isOpened():
        print('Video is not opened! {}'.format(videoins))
    else:  
        fps = vvv.get(cv2.CAP_PROP_FPS)  
        totalFrameNumber = vvv.get(cv2.CAP_PROP_FRAME_COUNT)
        size = (int(vvv.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vvv.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
        second = totalFrameNumber//fps

        if totalFrameNumber != 0:
            for _ in range(int(totalFrameNumber)):
                rval, frame = vvv.read()   
                if frame is not None: 
                    img = Image.fromarray(frame.astype('uint8')).convert('RGB')
                    imgtrans = centrans(img).numpy()                 
                    allframes.append(imgtrans)  

    return np.array(allframes)




if __name__ == "__main__":
    maxlen = 2000                                           # the maximum number of video frames that GPU can process
    savepath = 'path_to_save_visual_features'
    datapath = 'path_to_input_videos'
    os.chdir(datapath)
    allvideos = os.listdir()
    allvideos.sort()
    centrans = transform_center()


    # load CLIP pre-trained parameters
    device = 'cuda'
    clip_model = load_clip_cpu('ViT-B-16')
    clip_model.to(device)
    for paramclip in clip_model.parameters():
        paramclip.requires_grad = False


    for vid in range(len(allvideos)):
        vidone = get_videos(allvideos[vid], datapath)      # shape = (T,3,224,224)
        print('transform %d video has been done!' % vid)

        vidinsfeat = []      
        for k in range(int(len(vidone)/maxlen)+1):         # if the video is too long, split the video
            segframes = torch.from_numpy(vidone[k*maxlen:(k+1)*maxlen]).to(device)
            vis_feats = clip_model.encode_image(segframes)
            vidinsfeat = vidinsfeat + vis_feats.cpu().numpy().tolist()
        vidinsfeat = np.array(vidinsfeat)                  # shape = (T,512)

        assert(len(vidinsfeat) == len(vidone))
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        np.save(savepath+allvideos[vid][:-4]+'.npy', vidinsfeat)

        print('visual features of %d video have been done!' % vid)

    print('all %d visual features have been done!' % len(allvideos))
