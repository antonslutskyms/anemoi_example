FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:33


COPY anemoi.yaml .

RUN conda env update -n pcta -f anemoi.yaml
