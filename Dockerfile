# docker build -t cpp-project .
# docker run -it --rm -v "$(pwd):/home/ubuntu" -e POLYGON_API_KEY=<api_key> cpp-project /bin/bash

# Use an official Ubuntu as a base image
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    gnupg2 \
    cmake \ 
    libssl-dev \
    git

# Add GCC repository and install GCC 13.2.0
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-13 g++-13

# Set GCC 13 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# Copy the contents of the local repository to /ubuntu/home in the image
COPY . /home/ubuntu

# Make the entrypoint script executable
RUN chmod a+x /home/ubuntu/entrypoint.sh

# Set entrypoint to run the script with user input as argument
ENTRYPOINT ["/home/ubuntu/entrypoint.sh"]

# Set working directory
WORKDIR /home/ubuntu/

# Verify GCC version
RUN gcc --version

