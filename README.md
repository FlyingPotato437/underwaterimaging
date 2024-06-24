# Underwater Image Correction Algorithm

This repository contains an underwater image correction algorithm that combines techniques from the SeaThru paper with neural network-generated depth maps from Monodepth2 developed for the Australian Insitute of Marine Science by Srikanth Samy. The hybrid approach that was taken here aims to significantly enhance the quality of underwater images by effectively utilizing depth information from underwater pictures.

## Prerequisites

To use this project, you need to clone the Monodepth2 repository and download the required models.

### Clone Monodepth2 Repository

Clone the Monodepth2 repository using the following command:

```sh
git clone https://github.com/nianticlabs/monodepth2.git
