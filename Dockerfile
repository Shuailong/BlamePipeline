FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.28
MAINTAINER Shuailong Liang <liangshuailong@gmail.com>

# RUN git clone https://github.com/Shuailong/BlamePipeline.git && cd BlamePipeline && pip install -r requirements.txt && python setup.py install

ADD ./ /BlamePipeline/
RUN cd BlamePipeline && pip install -r requirements.txt && python setup.py develop
