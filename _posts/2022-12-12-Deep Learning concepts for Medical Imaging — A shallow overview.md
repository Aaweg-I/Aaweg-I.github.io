# Deep Learning concepts for Medical Imaging — A shallow overview

U-Net Architecture (src: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597))

In this article I go through a few DL concepts used in Medical Imaging. This is a very limited piece, and depends highly on (can be considered a limited summary of) this paper: \[https://arxiv.org/abs/1702.05747]. It is a non-mathematical, non-rigorous treatment, with focus on uses and concepts involved.

I will go through the use of deep learning for

1.  image classification,
2.  object detection,
3.  segmentation,

and provide concise overviews of conceptual ideal per application area.

For a fuller treatment, one must go through the article mentioned.

Let’s jump onto it.

# **Image Classification**

Image or exam classification was one of the first areas in which deep learning contributed majorly to medical image analysis. In exam classification, one typically has one or multiple images (an exam) as input, with a single diagnostic variable as output (e.g., disease present or not).

In such a setting, every diagnostic exam is a sample and dataset sizes are typically small compared to those in computer vision (e.g., hundreds/thousands vs. millions of samples). The popularity of transfer learning for such applications is therefore not surprising.

Transfer learning is the use of pre-trained networks (typically on natural images) to work around the (perceived) requirement of large data sets for deep network training.

Two transfer learning strategies were identified: (1) using a pre-trained network as a feature extractor and (2) fine-tuning a pre-trained network on medical data. The former strategy has the extra benefit of not requiring one to train a deep network at all, allowing the extracted features to be easily plugged in to existing image analysis pipelines. Both strategies are popular and have been widely applied. However, few authors perform a thorough investigation in which strategy gives the best result.

The medical imaging community initially focused on unsupervised pre-training and network architectures, like SAEs and RBMs, for classification. But recently, a clear shift towards CNNs can be observed. Authors also leverage unique attributes of medical data. Like using 3D convolutions instead of 2D. New layers have been developed, like edge-to-edge, edge- to-node, and node-to-graph layers.

CNNs are the current standard techniques. Especially CNNs pre-trained on natural images indicated strong results, challenging the accuracy of human experts in some tasks. Authors have also shown that CNNs can be adapted to leverage the intrinsic structure of medical images.

# **Object (or lesion) Classification.**

Object classification focuses on the classification of a small (previously identified) part of the medical image into two or more classes (e.g., Opacity classification in RSNA CXR). For such tasks, both local information on lesion appearance and global contextual information on lesion location are required for accurate classification. This combination is typically not possible in generic deep learning architectures. So, several authors have used multi-stream architectures to resolve this in a multi-scale fashion.

Incorporating 3D information is also often a necessity for good performance in object classification tasks in medical imaging. As images in computer vision are 2D natural images, networks developed in those scenarios do not directly leverage 3D information. Authors have used different approaches to integrate 3D effectively with custom architectures.

In all cases, almost all recent papers prefer the use of ‘end-to-end’ trained CNNs. An interesting approach, especially where object annotation to generate training data is expensive, is the integration of multiple instance learning (MIL) and deep learning, which appears to be superior to handcrafted features, and closely approaches the performance of a fully supervised method. We expect such approaches to be popular in the future as well, as obtaining high-quality annotated medical data is challenging.

Overall, object classification sees less use of pre-trained networks compared to exam classifications, mostly because of the need for incorporation of contextual or three-dimensional information. Several authors have found innovative solutions to add this information to deep networks with good results, and we expect deep learning to become even more prominent for this task in the coming times.

# **Object or lesion detection.**

Anatomical object localization (in space or time), such as organs or landmarks, has been an important preprocessing step in segmentation tasks or in the clinical workflow for therapy planning and intervention. And localization through 2D image classification with CNNs seems to be the most popular strategy to identify organs, regions and landmarks, with good results.

In fact, the detection of objects of interest or lesions in images is a key part of diagnosis and is one of the most labor-intensive tasks for clinicians. It comprises localization and identification of small lesions in the full image space. There has been a long research tradition in computer-aided detection systems that can detect lesions automatically, improving the detection accuracy or decreasing the reading time of human experts.

Most of the published deep learning object detection systems still use CNNs to perform pixel (or Voxel) classification, after which some form of post processing is applied to obtain object candidates. As the classification task performed at each pixel is essentially object classification, CNN architecture and methodology are like those in case of Classification. The incorporation of contextual or 3D information is handled using multi-stream CNNs. But also, as the annotation burden to generate training data can be similarly significant compared to object classification, some authors have looked towards weakly supervised deep learning.

# **Object Detection and Object Classification**

There are some aspects which are significantly different between object detection and object classification.

One key point is that because every pixel is classified, the class balance is typically skewed severely towards the non-object class in a training setting. To add insult to injury, most of the non-object samples are easy to discriminate, preventing the deep learning method from focusing on the challenging samples. van Grinsven et al. (2016) proposed a selective data sampling in which wrongly classified samples were fed back to the network more often to focus on challenging areas in retinal images. Last, as classifying each pixel in a sliding window fashion results in orders of magnitude of redundant calculation, fCNNs are important aspects of an object detection pipeline as well.

Challenges in meaningful application of deep learning algorithms in object detection are thus mostly similar to those in object classification. Only a few papers directly address issues specific to object detection- like class imbalance/hard-negative mining or efficient pixel/voxel-wise processing of images. We expect that more emphasis will be given to those areas in the near future, for example in the application of multi-stream networks in a fully convolutional fashion.

# **Segmentation**

The segmentation of organs and other substructures in medical images allows quantitative analysis of clinical parameters related to volume and shape, as, for example, in cardiac or brain analysis. Furthermore, it is often an important first step in computer-aided detection pipelines. Segmentation is defined as identifying the set of voxels which make up either the contour or the interior of the object (s) of interest. Segmentation is the most common subject of papers applying deep learning to medical imaging, and, as such, has also seen the widest variety in methodology, including the development of unique CNN-based segmentation architectures and the wider application of RNNs.

The most well-known, in medical image analysis, of these novel CNN architectures is U-net, published by Ronneberger et al. (2015). The two main architectural novelties in U-net are the combination of an equal amount of up-sampling and down-sampling layers. Although learned up sampling layers have been proposed before, U-net combines them with skip connections between opposing convolution and deconvolution layers. This which concatenate features from the contracting and expanding paths. From a training perspective, this means that an entire set of images/scans can be processed by U-net in one forward pass, resulting in a segmentation map directly. This allows U-net to consider the full context of the image, which can be an advantage in contrast to patch-based CNNs.

Although these specific segmentation architectures offered compelling advantages, many authors have also obtained excellent segmentation results with patch-trained neural networks. One of the earliest papers covering medical image segmentation with deep learning algorithms used such a strategy and was published by Ciresan et al.(2012). They applied pixel-wise segmentation of membranes in electron microscopy imagery in a sliding window fashion. Most recent papers now use fCNN (fully connected Convolutional Neural Networks) over sliding- window-based classification to reduce redundant computation.

Summarizing, segmentation in medical imaging has seen a huge influx of deep learning related methods. Custom architectures have been created to target the segmentation task directly. These have given promising results, rivaling and often improving over results obtained with fCNNs.

# **Lesion segmentation**

Segmentation of lesions combines the challenges of object detection and organ and substructure segmentation in the application of deep learning algorithms. Global and local context are typically needed to perform accurate segmentation, such that multi-stream networks with different scales or non-uniformly sampled patches are used. In lesion segmentation, we have also seen the application of U-net and similar architectures to leverage both this global and local context. One other challenge that lesion segmentation shares with object detection is class imbalance, as most voxels/pixels in an image are from the non-diseased class. Some papers combat this by adapting the loss function: Brosch et al. (2016) defined it to be a weighted combi- nation of the sensitivity and the specificity, with a larger weight for the specificity to make it less sensitive to the data imbalance. Others balance the data set by performing data augmentation on positive samples.

Thus, lesion segmentation sees a mixture of approaches used in object detection and organ segmentation. Developments in these two areas will most likely naturally propagate to lesion segmentation as the existing challenges are also mostly similar.

# **References**

1.  “[A Survey on Deep Learning in Medical Image Analysis](https://arxiv.org/abs/1702.05747)” by  
    Geert Litjens, Thijs Kooi, Babak Ehteshami Bejnordi, Arnaud Arindra Adiyoso Setio, Francesco Ciompi, Mohsen Ghafoorian, Jeroen A.W.M. van der Laak, Bram van Ginneken, Clara I. Sánchez
