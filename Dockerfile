FROM rapidsai/miniforge-cuda:cuda11.5.2-base-ubuntu20.04-py3.10
RUN apt-get update
RUN apt-get install build-essential -y
RUN echo "Copying files"
COPY ./* /apps/
WORKDIR /apps
RUN echo "Building Environments"
RUN mamba env create -f environment.yml
RUN echo "Environment ready"
RUN echo "source activate gala" > ~/.bashrc
ENV PATH /opt/conda/envs/gala/bin:$PATH
RUN pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_cluster-1.5.8-cp37-cp37m-linux_x86_64.whl
RUN pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
RUN pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
RUN pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
RUN pip install torch_geometric==1.6.3
RUN pip install fair-esm
ENTRYPOINT ["python","train.py"]
