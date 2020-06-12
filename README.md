# famous_classifiers_from_scratch

Building famous deep classification networks from scratch in PyTorch and train them to state-of-the-art performance.

Evaluation

   Network                                  ImageNet Evaluation                                
 ------------ -------------------------------------------------------------------------------- 
  AlexNet      1. Resize image to 256x256.                                                     
               2. Take 224x224 crops from image center                                         
               and 4 corners, as well as, their horizonal flips (10 crops).                    
               3. These 10 crops' softmax outputs are averaged for prediction.                 
               
  VGG family   (a) Dense                                                                       
               1. Resize the shorter dimension of image to Q                                   
               2. Run network densely over the image as in Overfeat                            
               (mention Andrew Ng video and other references here)                             
               3. The prediction feature map is averaged spatially for prediction.             
               (b) Multi-crop                                                                  
               1. Resize the shorter dimension of the image to 2 scales (Q1, Q2, Q3).          
               2. For each scale, take regular 5x5 grid crops and their horizontal flips.      
               i.e. The input for network is 224x224. If the resized image is 512x384,         
               then every time we shift the crop location by the rounding of (512-224)/5=57.6  
               horizontally or (384-224)/5=32 vertically, starting from top-left corner.       
               (c) Combined                                                                    
               1. Average of the final (averaged) softmax probabilities from (a) and (b)       
               3. The 3x5x5x2=150 crops' softmax probabilities are averaged for prediction.    
               
  Inception    1. Resize the shorter dimension of the image to 256, 288, 320 and 352           
               2. Take left, center and right squares (for landscape image), or, top,          
               center and bottom squares (for portrait).                                       
               3. For each square take the 5 224x224 crops as in AlexNet and the whole         
               square resized to 224x224, as well as their horizontal flips (12 crops)         
               4. The 4x3x12 = 144 crops' softmax probabilities are averaged for prediction.
               
  ResNet       (a) 10-crop evaluation as in AlexNet                                            
               (b) Average the dense (as in VGG (a) above) probabilities for                   
               multiple scales, where the images are resized with shorter                      
               dimension taking 224, 256, 384, 480, 640                                        
               
  *Note        The model ensemble evaluation is ignored in this table.    
