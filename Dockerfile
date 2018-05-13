FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.28
MAINTAINER Shuailong Liang <liangshuailong@gmail.com>

ADD ./ /BlamePipeline/
RUN cd BlamePipeline && pip install -r requirements.txt && python setup.py develop
# RUN download data files
# RUN prepare data for all submodules