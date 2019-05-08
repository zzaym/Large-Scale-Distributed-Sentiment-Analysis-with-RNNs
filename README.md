# Large-Scale-Distributed-Sentiment-Analysis-With-RNNs

Harvard CS205 Final Project, Spring 2019

Jianzhun Du, Rong Liu, Matteo Zhang, Yan Zhao

# Introduction 

In the current era, social medias are so common that people are constantly expressing their feelings through text. There are tremendous business values underlying this information. **Therefore, we hope to perform sentiment analysis with Recurrent Neural Networks (RNN) in order to uncover whether a piece of text has positive or negative sentiment.** 

This is a big data and big compute combined problem. It involves big data because in our selected dataset, we handle 92.45 GB of 142.8 million reviews. It involves big compute because during the training process of RNN, we need to frequently calculate the loss and update the parameters. Moreover, because of the nature of natural language processing, the vector representations of the text has very high dimensionality.

We employ MapReduce on AWS cluster to first preprocess the large amount of data. The mapper processes the input line by line, and the reducer combines the sorted intermediate output. HDF5 file format has been used to load the data without blowing up the memory. After processing the data, we use an AWS GPU cluster to distribute the workload across multiple GPUs by using large minibatch technique to speed up the RNN training on Pytorch and using its MPI interface with NCCL backend for communication between nodes. 

We demonstrate the details of this project in [here](https://sophieyanzhao.github.io). 


 



# How to use?

Please first download the source code in this repo, then take 0.5~1 hour (WARNING) to follow this reproduction instruction.

## I. Data Preprocessing 

### I.1 MapReduce

#### Uploading Files to S3 Bucket

Create a new bucket with no special setting and upload the following files:

- `install_boto3_h5py.sh`: installs boto3 and h5py packages on each node in the cluster
- `complete.json`: raw data
- `mapper.py` and `reducer.py`: python files for the MapReduce task

#### Deploying CPU Cluster on AWS

Please go to EMR dashboard and select `Create cluster`, and then `Go to advanced options`.

- **Step 1: Software and Steps**: Choose `emr-5.8.0` Release and leave the rest as default

- **Step 2: Hardware**: At the bottom of the page, change instance type as `m4.xlarge` for Master and Core, and change to `8 Core Instances`, or however many needed

- **Step 3: General Cluster Settings**: Add `custom bootstrap action` by calling the script `install_boto3_h5py.sh`. This bash file will install boto3 and h5py packages on each node in the cluster. Could also rename cluster or enable logging, but these are not required

- **Step 4: Security**: Select `EC2 key pair` and create cluster

#### Running MapReduce

After the cluster is started and bootstrapped, go to `Steps` tab and `Add step`:

- **Step type**: `Streaming program`
- **Name**: optional
- **Mapper**: path to your mapper file, e.g.`s3://BucketName/mapper.py`
- **Reducer**: path to your reducer file, e.g.`s3://BucketName/reducer.py`
- **Input S3 location**: path to your data file, e.g.`s3://BucketName/complete.json`
- **Output S3 location**: input a non-existing folder name, e.g.`s3://BucketName/new_folder`


### I.2 Combine Generate h5 Files

#### Launching Instance

Please go to EC2 dashboard and select `Launch Instance`.

- **Step 1: Choose AMI**: Launch `Ubuntu Server 16.04 LTS (HVM), SSD Volume Type`

- **Step 2: Choose an Instance Type**: Choose `m4.2xlarge`

- **Step 3: Launch**: Launch with your own key pair, such as `course-key`.


#### Installing Essential Packages

After connecting to the instance, you need to first install boto3 and h5py with the following commands:

```sudo apt update```

`sudo apt install python-pip`

`pip install boto3`

`pip install h5py`

#### Modify Instance Volume

Go to the EC2 dashboard, select `Volumes`, and modify the instance volume to 64GB.

Run `sudo growpart /dev/xvda 1` on the instance and restart it.

If you run `df -h`, you should find that the disk space has been expanded.

#### Running Python File

Specify bucket name and the names of the files to be combined in `combine_h5.py`. You can find the file names from the S3 bucket page. You may also adjust the output file name.

Next, upload the python file to instance and run it with `python combine_h5.py`

#### Final Setting

Make the preprocessed h5 data file public, in order for the following process to access it with its `Object URL`.

Go to the S3 bucket, select the `Permissions` tab, and set all options under `Public access settings` to False.

Find the combined h5 file and `Make public` under the `Actions` tab.


## II. RNN with Distributed SGD

#### Deploying GPU Cluster on AWS

Please go to EC2 dashboard and select `Launch Instance`.

- **Step 1: Choose AMI**: Launch `Deep Learning AMI (Ubuntu) Version 22.0`

- **Step 2: Choose an Instance Type**: Choose g3.4xlarge for testing multiple nodes with 1 GPU each

- **Step 3: Configure Instance**: Select 4 (however much you need) for `Number of instances` and leave the rest as default

- **Step 4: Add Storage**: Select 200GiB General Purpose. Adjust the size depending on how big the dataset is.

- **Step 5: Add Tags**: Skip this.

- **Step 6: Configure Security Group**: Create a new security group and choose a name like `GPU-group` for keeping track of the security.

- **Step 7: Review**: Launch with your own key pair, such as `course-key`.

Additional configuration for the security group is required so that the nodes can communicate with each other.

Please go to `Security Groups` and edit the rules

- **Edit Inbound Rules**: Add rule with `all traffic` as `Type` and `Custom` with this security group (type in `GPU-group` and will show the security group id) as `Source`.

- **Edit Outbound Rules**: Same as inbound.

#### Environment Setup

This setup needs to be done for each node individually.

First, activate the pytorch environment.

`source activate pytorch_p36`

Install the latest Pytorch 1.1.

`conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

Install h5pickle (a wrapper of `h5py` for distributed computing)

`conda config --add channels conda-forge`

`conda install h5pickle`

Find the name of private IP of the node by running `ifconfig` (usually `ens3`) and export it to NCLL socket:

`export NCCL_SOCKET_IFNAME=ens3` (add to `.bashrc` to make this change permanent)

Upload the scripts to each node or git clone from the repository.

Also, upload the data to each node if running without NFS (Network File System) setup.

#### Getting the processed data

Download the data processed using MapReduce by executing: 

`wget https://s3.amazonaws.com/cs205amazonreview/combined_result_5class.h5` 

#### Running the sequential version (only need 1 node)

Run the following command on one node:

`python sequential_rnn.py --dir <Input Path> --batch <Batch Size> --lr <Learning Rate> --epochs <# Epochs> --workers <# Workers> --n_vocab 10003 --filename <Model Name> > log.out &`

where `<Input Path>` is the path to the input file, `<# Workers>` is the # of CPUs to load the data; `<Model Name>` is the name of the RNN model for saving; `log.out` is the output log file. 

One example would be:

`python sequential_rnn.py --dir ../data/combined_result_5class.h5 --batch 128 --lr 0.1 --epochs 10 --workers 8 --n_vocab 10003 --filename model_1n_1g > log_1n_1g_b128.out &`

##### Profiling the sequential version

If you want to profile the sequential code, please replace `python` with `python -m cProfile` in the command above, as shown below:

`python -m cProfile -o sequential.prof sequential_rnn.py --dir <Input Path> --batch <Batch Size> --lr <Learning Rate> --epochs <# Epochs> --workers <# Workers> --n_vocab 10003 --filename <Model Name> > log.out &`

In order to visualize the profiling result, please install sneakviz by executing:

`pip install snakeviz`

Visualize the profiling by running:

`snakeviz sequential.prof`

## Running the distributed version 

Note this is the command for running code where each node keeps a local copy of the data.

For each node run:

```python -m torch.distributed.launch --nproc_per_node=<#GPU per Node> --nnodes=<Total # of Nodes> --node_rank=<i> --master_addr="<Master Node Private IP>" --master_port=<Free Port> main.py --dir <Input Path> --epochs <# Epochs> --workers <# Workers> --n_vocab <# Words in Dictionary> --dynamic <Dyanmic Mode> --filename <Model Name> > log.out &```

where `<#GPU per Node>` is the number of GPUs in each node; `<Total # of Nodes>` is the total number of nodes; `<i>` is the rank assigned to this node (starting from 0 = master node); `<Master Node Private IP>` is the private IP of master node which can be found by running `ifconfig` under `ens3`; `<Free Port>` is any free port; `<Input Path>` is the path to the input file; `<# Workers>` is the # of CPUs to load the data; `<# Words in Dictionary` is the number of words in the dictionary, and in our case it's 10003;`<Model Name>` is the name of the RNN model for saving; `Dynamic Mode` refers to using the dynamic load balancer, where negative value is not running, 0 runs only once after the 1st epoch and any positive real integer `j` updates the load every `j` epochs; `log.out` is the output log file. 

For example, running 2 nodes without dynamic load balancing and on the background and with log files would be:

Node 1: 

```python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.31.35.159" --master_port=23456 dynamic_rnn.py --dir ../data/combined_result_5class.h5  --batch 128 --lr 0.1 --epochs 10 --dynamic -1 --workers 8 --n_vocab 10003 --filename model_2n_1g > log.out &```

Node 2:

```python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="172.31.35.159" --master_port=23456 dynamic_rnn.py --dir ../data/combined_result_5class.h5  --batch 128 --lr 0.1 --epochs 10 --dynamic -1 --workers 8 --n_vocab 10003 --filename model_2n_1g > log.out &```

#### Configure NFS for file sharing

This is inspired by `Harvard CS205 - Spring 2019 - Infrastructure Guide - I7 - MPI on AWS`, but with modifications to bypass the extra user account created, which is unnecessary in this setting.

Let `master$` denote master node and `$node` denote any other node

Run the following commands on master node:

- Install NFS server: `master$ sudo apt-get install nfs-kernel-server`

- Create NFS directory: `master$ mkdir cloud`

- Export cloud directory: add the following line `/home/ubuntu/cloud *(rw,sync,no_root_squash,no_subtree_check)` to `/etc/exports` by executing `master$ sudo vi /etc/exports`

- Update the changes: `master$ sudo exportfs -a`

Configure the NFS client on other nodes:

- Install NFS client: `node$ sudo apt-get install nfs-common`

- Create NFS directory: `node$ mkdir cloud`

- Mount the shared directory: `node$ sudo mount -t nfs <Master Node Private IP>:/home/ubuntu/cloud /home/ubuntu/cloud`

- Make the mount permanent (optional): add the following line `<Master Node Private>:/home/ubuntu/cloud /home/ubuntu/cloud nfs` to `/etc/fstab` by executing `node$ sudo vi /etc/fstab
`

#### Running with NFS mounted directory

Please upload the data to NFS mounted directory `cloud` first.

For each node run:

```python -m torch.distributed.launch --nproc_per_node=<#GPU per Node> --nnodes=<Total # of Nodes> --node_rank=<i> --master_addr="<Master Node Private IP>" --master_port=<Free Port> main.py --dir <Mounted Input Path> --epochs <# Epochs> --workers <# Workers> --n_vocab <# Words in Dictionary> --dynamic <Dyanmic Mode> --filename <Model Name> > log.out &```

where `<#GPU per Node>` is the number of GPUs in each node; `<Total # of Nodes>` is the total number of nodes; `<i>` is the rank assigned to this node (starting from 0 = master node); `<Master Node Private IP>` is the private IP of master node which can be found by running `ifconfig` under `ens3`; `<Free Port>` is any free port; `<Mounted Input Path>` is the path to the mounted input file; `<# Workers>` is the # of CPUs to load the data; `<# Words in Dictionary` is the number of words in the dictionary, and in our case it's 10003;`<Model Name>` is the name of the RNN model for saving; `Dynamic Mode` refers to using the dynamic load balancer, where negative value is not running, 0 runs only once after the 1st epoch and any positive real integer `j` updates the load every `j` epochs; `log.out` is the output log file. 

For example, running 2 nodes without dynamic load balancing and on the background and with log files would be:

Node 1: 

```python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.31.35.159" --master_port=23456 dynamic_rnn.py --dir ../cloud/combined_result_5class.h5  --batch 128 --lr 0.1 --epochs 10 --dynamic -1 --workers 8 --n_vocab 10003 --filename model_2n_1g > log.out &```

Node 2:

```python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="172.31.35.159" --master_port=23456 dynamic_rnn.py --dir ../cloud/combined_result_5class.h5  --batch 128 --lr 0.1 --epochs 10 --dynamic -1 --workers 8 --n_vocab 10003 --filename model_2n_1g > log.out &```


## System Information

### Dependencies

- Python 3.6.5
- torch 1.1.0
- h5py 2.8.0 (we also use h5pickle, which is a wrapper of h5py)
- boto3 1.9.143


### GPU Instances Information

|     | GPUs | vCPU | Mem(GiB) | GPU Memory(GiB) | Max Bandwidth(Mbps) | Max Throughput(MB/s 128 KB I/O) | Maximum IOPS(16KB I/O) | GPU Card         | 
|-------------|------|------|----------|-----------------|---------------------|---------------------------------|------------------------|------------------| 
| p2.xlarge   | 1    | 4    | 61       | 12              | 750                 | 93.75                           | 6000                   | NVIDIA Tesla K80 | 
| g3.4xlarge  | 1    | 16   | 122      | 8               | 3500                | 437.5                           | 20000                  | NVIDIA Tesla M60 | 
| g3.16xlarge | 4    | 64   | 488      | 32              | 14000               | 1750                            | 80000                  | NVIDIA Tesla M60 | 

### CPU Instance on AWS EMR

|            | vCPUs      | Model Name                              | Memory(L2)       | Operating System   |                   
|------------|------------|-----------------------------------------|------------------|--------------------| 
| m4.xlarge  | 4          | Inter(R) Xeon(R) CPU E5-2686 v4 2.3GHz  | 256K             | Ubuntu 16.04.5 LTS | 
| m4.x2large | 8          | Intel(R) Xeon(R) CPU E5-2686 v4 2.30GHz | 256K             | Ubuntu 16.04.5 LTS | 



### CUDA Information
![p](cuda_info.png)
