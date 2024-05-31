# Purpose
This research aims to develop a rapid and efficient methodology for latent fingerprint recognition, addressing the challenge of high processing times in current approaches. The study explores the potential of one-shot and few-shot learning techniques in achieving accurate and computationally efficient latent fingerprint recognition.

# Data
The data is taken from the IIIT-D latent fingerprint dataset. The dataset contains 850 images of 3 groups. These 3 groups are IIITD Latent Mated 500ppi, IIITD Latent Mated 1000ppi, and IIITDLatentDatabase. The proposed solution is mainly on IIITD Latent Mated 1000ppi containing 254 images.

![image](https://github.com/nithinreddy003/latent_fingerprint/assets/104730933/f4dd0044-af3c-45ef-9eb1-aec2326078bd)
![image](https://github.com/nithinreddy003/latent_fingerprint/assets/104730933/cb5e2eeb-a716-45ee-a597-fe65364fb6ac)

# Architecture
![prototypical_network_architecture](https://github.com/nithinreddy003/latent_fingerprint/assets/104730933/2a7a8421-0825-4ee2-9c33-bf09f0de2fe2)



# Results
In the few-shot learning of the prototypical network using the pre-trained DenseNet121 model, the research achieved a remarkable test accuracy of 91.66%. Additionally, both the F1 score and precision metrics are 93.32% & 93.93%, indicating the high performance of the proposed methodology in latent fingerprint recognition.

![fine_tuned_200epoch](https://github.com/nithinreddy003/latent_fingerprint/assets/104730933/df87bc27-15e3-4ee1-bdb8-5bfc219db8dd)
![confusion_matrix_test_cases](https://github.com/nithinreddy003/latent_fingerprint/assets/104730933/4859b6b3-6792-4570-924a-7e232491dc11)
