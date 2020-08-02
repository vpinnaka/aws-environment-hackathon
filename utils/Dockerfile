# rapids nightly image
FROM rapidsai/rapidsai-nightly:0.15-cuda10.1-runtime-ubuntu18.04-py3.7

# update
RUN apt update && apt -y upgrade

# install pytorch
RUN source activate rapids && conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

