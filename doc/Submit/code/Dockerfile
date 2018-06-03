# Base cuda90
FROM allennlp/allennlp:v0.4.3
RUN pip install allennlp
RUN pip install git+https://github.com/pytorch/tnt.git@master

# install source code
ADD ./ /root/BlamePipeline/
WORKDIR /root/BlamePipeline/
RUN pip install -r requirements.txt && python setup.py develop
