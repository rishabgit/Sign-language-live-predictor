# Problem Statement 6

The problem statement demands a means of communication between two persons, one of whom is deaf and dumb. The process requires Sign Language which is assumed to be known by the deaf and dumb person but absolutely unknown to the other. The app should aid as a translator. Whatever the normal person says must be mapped to the sign language by the app OR whatever sign the differently abled person generates must be converted into proper audio.  

The system should be able to generate gestures for voice.  
The system should be able to generate voice for gestures.  
  
  
## Dataset
Download https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset and move each ASL class's folder (currently 28 classes) inside data/training/.                                                                                 
For the next dataset, run  
`python open_images_downloader.py --root ~/data/open_images --class_names "Human hand" --num_workers 20`  
which uses https://storage.googleapis.com/openimages/web/index.html for labeled dataset. Refer to  qfgaohao's https://github.com/qfgaohao/pytorch-ssd for more information.


References
https://arxiv.org/abs/1512.02325 - https://github.com/qfgaohao/pytorch-ssd                                                                 
https://arxiv.org/abs/1512.03385                                                                                                           
https://www.fast.ai/
