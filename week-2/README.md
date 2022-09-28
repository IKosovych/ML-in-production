
# Setup 

Create kind cluster 

```
python3 -m venv env
source env/bin/activate
pip install -r minio_part/requirements.txt
```
In case of error
```
Building wheel for aiohttp (PEP 517) ... error
```
You need to upgrade pip
```
pip install --upgrade pip
```

# MINIO 

Deploy 

```
kubectl create -f ./minio/minio-standalone.yaml
```


Access UI and API 

```
sudo docker run \
   -p 9000:9000 \
   -p 9090:9090 \
   --name minio1 \
   -v ~/dvc_part/data:/data \
   -e "MINIO_ROOT_USER=ROOTNAME" \
   -e "MINIO_ROOT_PASSWORD=CHANGEME123" \
   quay.io/minio/minio server /data --console-address ":9090"
```

# MINIO Client 


```
pytest test_minio_client.py
```
