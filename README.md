# Mlflow on AWS


### AWS setup:
1. Loging to AWS console.
2. Create IAM user with `AdministratorAccess`.
3. Export the credentials in you AWS CLI by running `"aws configure"`.
4. Create S3 bucket
5. Create EC2 (`linux`) machine and add the security groups `5000` port.

Running the follwoing command on EC2 machine
```bash
sudo apt udate

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdri mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell
```

### Configure the AWS-cli
aws configure

### Finally configure the mflow
mlflow server -h 0.0.0.0 --default-artifact-root s3://[s3-bucket-name]

###
Open the Public IPv4 DNS to the port `5000`

_Export the traking uri_
```bash
export MLFLOW_TRACKING_URI=http://ec2-3-86-151-183.compute-1.amazonaws.com:5000/
```